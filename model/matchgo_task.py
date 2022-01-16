# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
import logging
import os
import sys
from argparse import Namespace
from functools import lru_cache
from .bert_dictionary import BertDictionary
from fairseq.data.dictionary import Dictionary
import numpy as np
from fairseq import metrics, options, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
    FairseqDataset
)
from .language_pair_dataset import LanguagePairDataset
from .custom_util import fn_timer, show_memory_info
from fairseq.tasks import LegacyFairseqTask, register_task
import gc

EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)


def load_langpair_dataset(
        data_path,
        split,
        src,
        src_dict,
        tgt,
        tgt_dict,
        combine,
        dataset_impl,
        upsample_primary,
        left_pad_source,
        left_pad_target,
        max_source_positions,
        max_target_positions,
        prepend_bos=False,
        load_alignments=False,
        truncate_source=False,
        append_source_id=False,
        num_buckets=0,
        shuffle=True,
        pad_to_multiple=1,
        img2ids_path=None,
        img2vec_path=None,
        sku2vec_path=None,
        sku2vec_dict=None,
        args=None
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    sku_vec = data_utils.load_indexed_dataset(sku2vec_path, sku2vec_dict, dataset_impl)

    # if truncate_source:
    #     sku_vec = AppendTokenDataset(
    #         TruncateDataset(
    #             StripTokenDataset(sku_vec, src_dict.eos()),
    #             max_source_positions - 1,
    #         ),
    #         src_dict.eos(),
    #     )
    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        sku_vec = PrependTokenDataset(sku_vec, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        sku_vec = AppendTokenDataset(
            sku_vec, src_dict.index("[{}]".format(sku2vec_path))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

    if args.is_multi_files:
        if split in ('valid', 'test'):
            img2vec_path = '/'.join(
                data_path.split('/')[:-2] + ['bpe_mg_batch/image_patch_vectors_' + split + '.npy'])
            img2ids_path = '/'.join(data_path.split('/')[:-2] + ['bpe_mg_batch/' + split + '.img2ids'])
        else:
            part_num = int(data_path.split('part')[-1])
            img2vec_path = '/'.join(
                data_path.split('/')[:-2] + ['bpe_mg_batch/image_patch_vectors_part_' + str(part_num) + '.npy'])
            img2ids_path = '/'.join(
                data_path.split('/')[:-2] + ['bpe_mg_batch/part_' + str(part_num) + '.img2ids'])

    img_vec = IndexedImgDataset.instance()
    img_vec.read_data(img2vec_path=img2vec_path, img2ids_path=img2ids_path)

    return LanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
        img_vec=img_vec,
        sku_vec=sku_vec
    )


@register_task("matchgo")
class MatchgoTask(LegacyFairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner; \
                            however, valid and test data are always in the first directory to \
                            avoid the need for repeating them in all directories')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--load-alignments', action='store_true',
                            help='load the binarized alignments')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--truncate-source', action='store_true', default=False,
                            help='truncate source to max-source-positions')
        parser.add_argument('--num-batch-buckets', default=0, type=int, metavar='N',
                            help='if >0, then bucket source and target lengths into N '
                                 'buckets and pad accordingly; this is useful on TPUs '
                                 'to minimize the number of compilations')

        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenize before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')
        # fmt: on
        parser.add_argument(
            "--sku2vec-path",
            type=str,
            default=None,
            help="sku2vec file path",
        )
        parser.add_argument(
            "--img2ids-path",
            type=str,
            default=None,
            help="img ids file path",
        )
        parser.add_argument(
            "--img2vec-path",
            type=str,
            default=None,
            help="img vec file path",
        )
        parser.add_argument('--bertdict', action='store_true', default=False,
                            help='use bert dictionary')

        parser.add_argument(
            "--task_type",
            choices=["kplug", "matchgo", "tpg", "vpg", "single_vpg", "vpg_test", "new_vpg", "new_single_vpg",
                     'new_vpg_test', 'new_tpg', 'vpg_none'],
            default="matchgo",
            help="try different tasks",
        )
        parser.add_argument(
            "--patch_embed_size",
            type=int,
            default=1024,
        )
        parser.add_argument(
            "--patch_num",
            type=int,
            default=196,
        )
        parser.add_argument(
            "--is_multi_files",
            action='store_true',
            default=False,
        )
        parser.add_argument(
            "--add_rel_margin",
            type=int,
            default=0,
        )
        parser.add_argument(
            "--rel_margin",
            type=float,
            default=1.0,
        )

    def __init__(self, args, src_dict, tgt_dict, sku2vec_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.sku2vec_dict = sku2vec_dict

    @classmethod
    def load_dictionary(cls, filename, bertdict=False):
        if bertdict:
            return BertDictionary.load_from_file(filename)
        return Dictionary.load(filename)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        for key, item in vars(args).items():
            print("k: {}, v: {}.".format(key, item))

        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(
                paths[0]
            )
        if args.source_lang is None or args.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(args.source_lang)),
            bertdict=args.bertdict
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(args.target_lang)),
            bertdict=args.bertdict
        )
        """
        sku2vec_dict = cls.load_dictionary(
            args.sku2vec_path
        )
        """
        sku2vec_dict = src_dict
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(args.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict, sku2vec_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        if split != getattr(self.args, "train_subset", None):
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
            num_buckets=self.args.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.args.required_seq_len_multiple,
            img2ids_path=self.args.img2ids_path + '/' + split + '.img2ids',
            img2vec_path=self.args.img2vec_path,
            sku2vec_path=self.args.sku2vec_path + '/' + split + '.sku2vec',
            sku2vec_dict=self.sku2vec_dict,
            args=self.args,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )

    def build_model(self, args):
        # from fairseq import pdb; pdb.set_trace()
        model = super().build_model(args)
        if getattr(args, "eval_bleu", False):
            assert getattr(args, "eval_bleu_detok", None) is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(getattr(args, "eval_bleu_detok_args", "{}") or "{}")
            self.tokenizer = encoders.build_tokenizer(
                Namespace(
                    tokenizer=getattr(args, "eval_bleu_detok", None), **detok_args
                )
            )

            gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.args.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])


class IndexedImgDataset(FairseqDataset):
    """Takes img ids and img vector files as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    ins_ = None

    @classmethod
    def instance(cls):
        cls.ins_ = cls.ins_ if cls.ins_ else IndexedImgDataset()
        return cls.ins_

    def __init__(self):
        self.tokens_list = []
        self.sizes = []
        self.lines = []

        self.size = len(self.tokens_list)

    @fn_timer
    def read_data(self, img2vec_path, img2ids_path):
        # way-1
        # self.tokens_list = []
        # self.sizes = []
        # self.lines = []

        # way-2
        # self.tokens_list[:] = []
        # self.sizes[:] = []
        # self.lines[:] = []

        # way-3
        # del self.tokens_list
        # del self.sizes
        # del self.lines

        # way-4
        show_memory_info("before gc torch image_feature ")
        self.image_feature = []

        print("[1]image_feature ref num: {}".format(sys.getrefcount(self.image_feature) - 1))
        self.tokens_list = []
        self.sizes = []
        self.lines = []

        print("[2]image_feature ref num: {}".format(sys.getrefcount(self.image_feature) - 1))
        show_memory_info("after gc torch image_feature ")

        show_memory_info("before load numpy image_feature")
        self.image_feature = np.load(img2vec_path)
        print("[3]image_feature ref num: {}".format(sys.getrefcount(self.image_feature) - 1))
        show_memory_info("after load numpy image_feature")

        print("[tokens_list] obj id: {}".format(id(self.tokens_list)))
        print("[sizes] obj id: {}".format(id(self.sizes)))
        print("[lines] obj id: {}".format(id(self.lines)))
        print("[image_feature] obj id: {}".format(id(self.image_feature)))
        print('################img2ids_path', img2ids_path)
        print('################img2vec_path: {}, shape is {}'.format(img2vec_path, np.shape(self.image_feature)))

        show_memory_info("before load torch image_feature ")
        with open(img2ids_path, "r", encoding="utf-8") as f:
            input_lines = f.readlines()
            max_len = max([len(line.strip().split()) for line in input_lines])
            print('max_len', max_len)

            for line in input_lines:
                self.lines.append(line.strip())
                id_list = [int(seg) for seg in line.strip().split()]
                img = None
                for idx in id_list:
                    if img is None:
                        img = self.image_feature[idx]
                    else:
                        img += self.image_feature[idx]
                    # cur_token.append(image_feature[idx])
                img /= len(id_list)
                # for _ in range(max_len - len(id_list)):
                # cur_token.append(np.zeros(feature_dim, dtype=np.float32))
                # self.tokens_list.append(cur_token)
                self.tokens_list.append(img)
                # self.sizes.append(len(id_list))
                self.sizes.append(1)
        show_memory_info("after load torch image_feature ")
        print("[4]image_feature ref num: {}".format(sys.getrefcount(self.image_feature) - 1))

        self.size = len(self.tokens_list)
        print('self.size', self.size)

    # def read_data(self, img2ids_path, img2vec_path):
    #     sku_path = str(img2ids_path).replace('img2ids', 'sku')
    #     image_feature_dir = "/".join(str(img2vec_path).split('/')[:-1] + ['image_patch_vectors'])
    #     print('################sku_path', img2ids_path)
    #     print('################image_feature_dir', img2vec_path)
    #
    #     with open(sku_path, "r", encoding="utf-8") as f:
    #         input_lines = f.readlines()
    #         for line in input_lines:
    #             self.lines.append(line.strip())
    #
    #             img = np.load(image_feature_dir + '/' + line.strip() + '.npy').squeeze()
    #
    #             self.tokens_list.append(img)
    #
    #             self.sizes.append(1)
    #     self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            print("i: ", i)
            print('self.size', self.size)
            print('self.tokens_list', len(self.tokens_list))
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)
