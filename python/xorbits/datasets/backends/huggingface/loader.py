# Copyright 2022-2023 XProbe Inc.
# derived from copyright 1999-2021 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections.abc
import inspect
import itertools
import logging
import os.path

from ...._mars.core.context import get_context
from ...._mars.core.entity import OutputType
from ...._mars.serialization.serializables import (
    BoolField,
    DictField,
    Int32Field,
    StringField,
)
from ...._mars.typing import OperandType
from ...operand import DataOperand, DataOperandMixin


logger = logging.getLogger(__name__)


class HuggingfaceLoader(DataOperand, DataOperandMixin):
    path = StringField("path")
    kwargs = DictField("kwargs")
    single_data_file = StringField("single_data_file")
    inspected = BoolField("inspected")
    num_blocks: int = Int32Field("num_blocks")
    block_index: int = Int32Field("block_index")
    cache_dir: str = StringField("cache_dir")
    split_urls = DictField("split_urls")

    def __call__(self):
        self.output_types = [OutputType.huggingface_dataset]
        return self.new_tileable([])

    @classmethod
    def _get_kwargs(cls, obj, kwargs):
        sig_builder = inspect.signature(obj)
        return {
            key: kwargs[key] for key in sig_builder.parameters.keys() if key in kwargs
        }

    @classmethod
    def _get_dl_manager(cls, builder, kwargs, dl_manager_cls):
        from datasets import DownloadConfig, DownloadMode, VerificationMode

        num_proc = kwargs.get("num_proc")
        download_config = kwargs.get("download_config")
        download_mode = kwargs.get(
            "download_mode", DownloadMode.REUSE_DATASET_IF_EXISTS
        )
        verification_mode = kwargs.get(
            "verification_mode", VerificationMode.BASIC_CHECKS
        )

        dl_manager = builder.dl_manager
        if dl_manager is None:
            if download_config is None:
                download_config = DownloadConfig(
                    cache_dir=builder._cache_downloaded_dir,
                    force_download=download_mode == DownloadMode.FORCE_REDOWNLOAD,
                    force_extract=download_mode == DownloadMode.FORCE_REDOWNLOAD,
                    use_etag=False,
                    num_proc=num_proc,
                    storage_options=builder.storage_options,
                )  # We don't use etag for data files to speed up the process

            dl_manager = dl_manager_cls(
                dataset_name=builder.name,
                download_config=download_config,
                data_dir=builder.config.data_dir,
                base_path=builder.base_path,
                record_checksums=(
                    builder._record_infos
                    or verification_mode == VerificationMode.ALL_CHECKS
                ),
            )
            builder.download_and_prepare()
        return dl_manager

    @classmethod
    def _inspect_split_urls(cls, builder, kwargs):
        from datasets import DownloadManager

        class InspectDownloadManager(DownloadManager):
            def download_and_extract(self, url_or_urls):
                return url_or_urls

            def download(self, url_or_urls):
                return url_or_urls

        try:
            dl_manager = cls._get_dl_manager(builder, kwargs, InspectDownloadManager)
            split_generators = list(builder._split_generators(dl_manager))
            split_to_urls = {}
            for sp in split_generators:
                maybe_urls = []
                for v in sp.gen_kwargs.values():
                    if isinstance(v, collections.abc.Sequence) and not isinstance(
                        v, str
                    ):
                        if all(isinstance(e, str) for e in v):
                            maybe_urls.append(v)
                if len(maybe_urls) == 1:
                    split_to_urls[sp.name] = maybe_urls[0]
            return split_to_urls
        except Exception as e:
            logger.debug("Inspect data urls failed: %s", e)
            return {}

    @classmethod
    def tile(cls, op: OperandType):
        assert len(op.inputs) == 0

        import datasets

        builder_kwargs = cls._get_kwargs(datasets.load_dataset_builder, op.kwargs)
        builder = datasets.load_dataset_builder(op.path, **builder_kwargs)
        data_files = builder.config.data_files
        # TODO(codingl2k1): check data_files if can be supported
        split = op.kwargs.get("split")
        # TODO(codingl2k1): support multiple splits

        if data_files and split and len(data_files[split]) > 1:
            split_urls = {}
            data_files = data_files[split]
            inspected = False
        else:
            split_urls = cls._inspect_split_urls(builder, op.kwargs)
            data_files = split_urls.get(split)
            inspected = True

        chunks = []
        if data_files:
            ctx = get_context()
            # TODO(codingl2k1): make expect worker binding stable for cache reuse.
            all_bands = [b for b in ctx.get_worker_bands() if b[1].startswith("numa-")]
            for index, (f, band) in enumerate(
                zip(data_files, itertools.cycle(all_bands))
            ):
                chunk_op = op.copy().reset_key()
                assert f, "Invalid data file from DatasetBuilder."
                chunk_op.single_data_file = f
                chunk_op.num_blocks = len(data_files)
                chunk_op.block_index = index
                chunk_op.cache_dir = builder.cache_dir
                chunk_op.expect_band = band
                chunk_op.inspected = inspected
                chunk_op.split_urls = split_urls
                c = chunk_op.new_chunk(inputs=[], index=index)
                chunks.append(c)
        else:
            chunk_op = op.copy().reset_key()
            chunk_op.single_data_file = None
            chunks.append(chunk_op.new_chunk(inputs=[]))

        return op.copy().new_tileable(op.inputs, chunks=chunks)

    @classmethod
    def execute(cls, ctx, op: OperandType):
        from datasets import (
            load_dataset_builder,
            DatasetBuilder,
            VerificationMode,
            DownloadManager,
        )

        builder_kwargs = cls._get_kwargs(load_dataset_builder, op.kwargs)

        # TODO(codingl2k1): not sure if it's OK to share one cache dir among workers.
        # if op.single_data_file:
        #     # TODO(codingl2k1): use xorbits cache dir
        #     new_cache_dir = os.path.join(op.cache_dir, f"part_{op.block_index}_{op.num_blocks}")
        #     builder_kwargs["cache_dir"] = new_cache_dir

        # load_dataset_builder from every worker may be slow, but it's error to
        # deserialized a builder instance in a clean process / node, e.g. raise
        # ModuleNotFoundError: No module named 'datasets_modules'.
        #
        # Please refer to issue: https://github.com/huggingface/transformers/issues/11565
        builder = load_dataset_builder(op.path, **builder_kwargs)
        download_and_prepare_kwargs = cls._get_kwargs(
            DatasetBuilder.download_and_prepare, op.kwargs
        )

        if op.single_data_file is not None:
            output_dir = builder._output_dir
            output_dir = output_dir if output_dir is not None else builder.cache_dir
            output_dir = os.path.join(
                output_dir, f"part_{op.block_index}_{op.num_blocks}"
            )
            download_and_prepare_kwargs["output_dir"] = output_dir
            download_and_prepare_kwargs[
                "verification_mode"
            ] = VerificationMode.NO_CHECKS
            split = op.kwargs["split"]
            if op.inspected:
                data_urls = op.split_urls[split]
                print("dddd", data_urls)

                class InspectSplitDownloadManager(DownloadManager):
                    def download(self, url_or_urls):
                        print("fffff", url_or_urls)
                        split_urls = {}
                        if isinstance(url_or_urls, collections.abc.Mapping):
                            for k, v in url_or_urls.items():
                                print(k, v == data_urls, v)
                                if v == data_urls:
                                    split_urls[k] = [op.single_data_file]
                                    for key in url_or_urls.keys():
                                        if key not in split_urls:
                                            split_urls[key] = []
                                    break
                        elif isinstance(url_or_urls, collections.abc.Sequence):
                            if url_or_urls == data_urls:
                                split_urls = [op.single_data_file]
                        print("vvvvvvvv", split_urls)
                        if split_urls:
                            return super().download(split_urls)
                        return super().download(url_or_urls)

                # TODO(codingl2k1): handle original dl_manager args.
                download_and_prepare_kwargs["dl_manager"] = cls._get_dl_manager(
                    builder, op.kwargs, InspectSplitDownloadManager
                )
            else:
                print("jjjjjjjjjjjjjjjjjjj")
                split_data_files = builder.config.data_files[split]
                split_data_files[:] = [op.single_data_file]

        builder.download_and_prepare(**download_and_prepare_kwargs)
        as_dataset_kwargs = cls._get_kwargs(DatasetBuilder.as_dataset, op.kwargs)
        ds = builder.as_dataset(**as_dataset_kwargs)
        ctx[op.outputs[0].key] = ds


def load_dataset_from_huggingface(path: str, **kwargs):
    op = HuggingfaceLoader(
        output_types=[OutputType.huggingface_dataset], path=path, kwargs=kwargs
    )
    return op()
