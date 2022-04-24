# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import re
import subprocess
from logging import getLogger
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from codegen_sources.model.src.evaluation.evaluator import EncDecEvaluator, EVAL_DATASET_SPLITS, SRC_ST_LANGS, \
    TARGET_ST_LANG, get_l1l2_string, eval_moses_bleu, EVAL_OBF_PROBAS
from codegen_sources.model.src.evaluation.subtoken_score import run_subtoken_score
from codegen_sources.model.src.trainer import get_programming_language_name
from codegen_sources.model.src.utils import (
    to_cuda,
    restore_segmentation,
    show_batch,
    add_noise,
    convert_to_text, TREE_SITTER_ROOT, read_file_lines,
)
from codegen_sources.preprocessing.lang_processors.lang_processor import LangProcessor

from iren.utils.ranking_metrics import run_ranking_scoring

# Adding obf_proba = 0 to evaluation (1 in the bottom not a bug)
EVAL_OBF_PROBAS.append(1)

logger = getLogger()


class DistillationEvaluator(EncDecEvaluator):
    def __init__(self, trainer, data, params):
        """
        Build encoder / decoder evaluator.
        """
        super().__init__(trainer, data, params)
        self.teacher_encoder = trainer.teacher_encoder
        self.teacher_decoder = trainer.teacher_decoder
        self.teacher_scores = {}
        self.t_hyp_paths = {}

    def evaluate_mt(
            self,
            scores,
            data_set,
            lang1,
            lang2,
            eval_bleu,
            eval_computation,
            eval_subtoken_score,
            span,
            deobfuscate=False,
            deobfuscate_probas=None,
    ):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        assert data_set in EVAL_DATASET_SPLITS
        assert lang1 in params.langs
        assert lang2 in params.langs
        rng = np.random.RandomState(0)
        torch_rng = torch.Generator().manual_seed(0)
        eval_st = params.eval_st
        if not params.is_master or "cl" in lang1:
            # Computing the accuracy on every node is useful for debugging but
            # no need to evaluate spend too much time on the evaluation when not on master
            eval_bleu = False
            eval_computation = False
            eval_subtoken_score = False
            eval_st = False

        # store hypothesis to compute BLEU score
        if params.eval_bleu_test_only:
            datasets_for_bleu = ["test"]
        else:
            datasets_for_bleu = [s for s in EVAL_DATASET_SPLITS if s != "train"]

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        self.eval_mode()
        encoder, decoder = self._get_enc_dec(self.encoder, self.decoder, lang2_id, params)
        teacher_encoder, teacher_decoder = self._get_enc_dec(self.teacher_encoder, self.teacher_decoder, lang2_id,
                                                             params)

        for deobfuscation_proba in (
                deobfuscate_probas if deobfuscate_probas is not None else [None]
        ):
            if deobfuscate:
                rng = np.random.RandomState(0)
            l1l2 = get_l1l2_string(lang1, lang2, deobfuscation_proba)
            eval_name = f"{data_set}_{l1l2}"

            n_words = 0
            xe_loss = 0
            n_valid = 0
            t_xe_loss = 0
            t_n_valid = 0
            hypothesis = []
            teacher_hypothesis = []
            sources = []
            references = []
            for i, batch in enumerate(
                    tqdm(self.get_iterator(
                        data_set, lang1, lang2 if lang2 != lang1 else None, span=span
                    ), desc=f"local-rank-{params.local_rank}-obf_proba-{deobfuscation_proba}", mininterval=1.)
            ):
                spans = None
                assert len(batch) >= 2
                if len(batch) == 2:
                    if lang1 == lang2:
                        x2, len2 = batch
                        x1, len1 = add_noise(
                            x2,
                            len2,
                            self.params,
                            len(self.data["dico"]) - 1,
                            rng,
                            torch_rng,
                        )
                    else:
                        (x1, len1, ids1, len_ids1), (x2, len2, ids2, len_ids2) = batch
                        if deobfuscate:
                            (x1, len1, x2, len2) = self.trainer.deobfuscate_by_variable(
                                x1, x2, deobfuscation_proba, params.roberta_mode, rng,
                                obf_type=params.obf_type,
                                shuffle_masks=params.shuffle_dobf_masks
                            )
                            if x1 is None:
                                continue
                else:
                    assert len(batch) == 3
                    (
                        (x1, len1, ids1, len_ids1),
                        (x2, len2, ids2, len_ids2),
                        (spans, len_spans, _, _),
                    ) = batch

                langs1 = x1.clone().fill_(lang1_id)
                langs2 = x2.clone().fill_(lang2_id)

                # target words to predict
                alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
                pred_mask = (
                        alen[:, None] < len2[None] - 1
                )  # do not predict anything given the last target word
                y = x2[1:].masked_select(pred_mask[:-1])
                assert len(y) == (len2 - 1).sum().item()

                # cuda
                x1, len1, langs1, x2, len2, langs2, y, spans = to_cuda(
                    x1, len1, langs1, x2, len2, langs2, y, spans
                )
                enc1, _, word_scores, loss = self._model_forward(encoder, decoder, langs1, langs2, len1, len2,
                                                                 pred_mask, spans, x1, x2, y)

                # update stats
                n_words += y.size(0)
                xe_loss += loss.item() * len(y)
                n_valid += (word_scores.max(1)[1] == y).sum().item()

                # generate translation - translate / convert to text
                if (
                        eval_bleu or eval_computation or eval_subtoken_score
                ) and data_set in datasets_for_bleu:
                    generated, lengths = self._generate_hypothesis(data_set, decoder, enc1, i, lang1, lang2, lang2_id,
                                                                   len1, params, x1, x2)

                    hypothesis.extend(
                        convert_to_text(
                            generated,
                            lengths,
                            self.dico,
                            params,
                            generate_several_reps=True,
                        )
                    )
                    references.extend(convert_to_text(x2, len2, self.dico, params))
                    sources.extend(convert_to_text(x1, len1, self.dico, params))

                    if eval_name not in self.t_hyp_paths:
                        t_enc1, _, t_word_scores, t_loss = self._model_forward(teacher_encoder, teacher_decoder,
                                                                               langs1, langs2, len1, len2,
                                                                               pred_mask, spans, x1, x2, y)
                        t_xe_loss += t_loss.item() * len(y)
                        t_n_valid += (t_word_scores.max(1)[1] == y).sum().item()
                        if (
                                eval_bleu or eval_computation or eval_subtoken_score
                        ) and data_set in datasets_for_bleu:
                            t_generated, t_lengths = self._generate_hypothesis(data_set, teacher_decoder, t_enc1, i,
                                                                               lang1, lang2, lang2_id, len1, params, x1,
                                                                               x2, is_teacher=True)
                            teacher_hypothesis.extend(
                                convert_to_text(
                                    t_generated,
                                    t_lengths,
                                    self.dico,
                                    params,
                                    generate_several_reps=True,
                                )
                            )

            # compute perplexity and prediction accuracy
            scores[f"{eval_name}_mt_ppl"] = np.exp(xe_loss / n_words)
            scores[f"{eval_name}_mt_acc"] = 100.0 * n_valid / n_words
            if f"{eval_name}_mt_ppl_teacher" not in self.teacher_scores:
                self.teacher_scores[f"{eval_name}_mt_ppl_teacher"] = np.exp(t_xe_loss / n_words)
                self.teacher_scores[f"{eval_name}_mt_acc_teacher"] = 100.0 * t_n_valid / n_words

            # write hypotheses
            if (
                    eval_bleu or eval_computation or eval_subtoken_score
            ) and data_set in datasets_for_bleu:
                if eval_name not in self.t_hyp_paths:
                    hyp_paths, self.t_hyp_paths[eval_name], ref_path, src_path = self.write_hypo_teacher_hypo_ref_src(
                        data_set,
                        hypothesis,
                        teacher_hypothesis,
                        lang1,
                        lang2,
                        params,
                        references,
                        scores,
                        sources,
                        deobfuscation_proba,
                    )
                else:
                    hyp_paths, ref_path, src_path = self.write_hypo_ref_src(
                        data_set,
                        hypothesis,
                        lang1,
                        lang2,
                        params,
                        references,
                        scores,
                        sources,
                        deobfuscation_proba,
                    )

            # check how many functions compiles + return same output as GT
            if eval_computation and data_set in datasets_for_bleu:
                print("compute_comp_acc")
                self.compute_comp_acc(
                    data_set,
                    hyp_paths,
                    hypothesis,
                    lang1,
                    lang2,
                    params,
                    ref_path,
                    scores,
                    roberta_mode=params.roberta_mode,
                )

            if (
                    eval_st
                    and data_set in datasets_for_bleu
                    and get_programming_language_name(lang1) == SRC_ST_LANGS
                    and get_programming_language_name(lang2) in TARGET_ST_LANG
            ):
                logger.info("Computing ST comp acc")
                self.compute_comp_acc(
                    data_set,
                    hyp_paths,
                    hypothesis,
                    lang1,
                    lang2,
                    params,
                    ref_path,
                    scores,
                    roberta_mode=params.roberta_mode,
                    evosuite_functions=True,
                )
            if eval_subtoken_score and data_set in datasets_for_bleu:
                self.compute_subtoken_metrics(scores, hyp_paths, ref_path, lang1, lang2, data_set, deobfuscation_proba)
                self.compute_subtoken_metrics(self.teacher_scores, self.t_hyp_paths[eval_name], ref_path, lang1, lang2,
                                              data_set, deobfuscation_proba, is_teacher=True)

            # compute BLEU score
            if eval_bleu and data_set in datasets_for_bleu:
                self.evaluate_bleu(scores, hyp_paths, ref_path, deobfuscation_proba, lang1, lang2, eval_computation,
                                   data_set)
                self.evaluate_bleu(self.teacher_scores, self.t_hyp_paths[eval_name], ref_path, deobfuscation_proba,
                                   lang1, lang2, eval_computation, data_set, is_teacher=True)

            if params.eval_ranking_metrics and params.beam_size > 1:
                self.compute_ranking_metrics(scores, hyp_paths, ref_path, deobfuscation_proba, lang1, lang2, data_set)
                self.compute_ranking_metrics(self.teacher_scores, self.t_hyp_paths[eval_name], ref_path,
                                             deobfuscation_proba, lang1, lang2, data_set, is_teacher=True)

            if (
                    deobfuscate
                    and eval_bleu
                    or eval_subtoken_score
                    and data_set in datasets_for_bleu
            ):
                # TODO clean lang1
                vizualize_do_files_student_teacher(lang1.split("_")[0], src_path, ref_path, hyp_paths,
                                                   self.t_hyp_paths[eval_name])
        scores.update(self.teacher_scores)

    @staticmethod
    def compute_ranking_metrics(scores, hyp_paths, ref_path, deobfuscation_proba, lang1, lang2, data_set,
                                is_teacher=False):
        prefix = f"{data_set}_{get_l1l2_string(lang1, lang2, deobfuscation_proba)}_mt_ranking"
        if is_teacher and any(
                key.startswith(prefix) and
                key.endswith("_teacher") for
                key in scores.keys()):
            return
        ranking_metrics = run_ranking_scoring(ref_path, hyp_paths)
        for metric_type, value in ranking_metrics.items():
            logger.info(
                f"Ranking {metric_type} score {hyp_paths} {ref_path} {'teacher' if is_teacher else ''}: {value:f}"
            )
            scores[
                f"{prefix}_{metric_type}" + ("_teacher" if is_teacher else "")
                ] = value

    @staticmethod
    def evaluate_bleu(scores, hyp_paths, ref_path, deobfuscation_proba, lang1, lang2, eval_computation, data_set,
                      is_teacher=False):
        prefix = f"{data_set}_{get_l1l2_string(lang1, lang2, deobfuscation_proba)}_mt_bleu"
        if is_teacher and any(
                key.startswith(prefix) and
                key.endswith("_teacher") for
                key in scores.keys()):
            return
        # evaluate BLEU score
        bleu = eval_moses_bleu(ref_path, hyp_paths[0])
        logger.info(f"BLEU {hyp_paths[0]} {ref_path} {'teacher' if is_teacher else ''}: {bleu:f}")
        scores[
            prefix + (f"_teacher" if is_teacher else "")
            ] = bleu
        if eval_computation:
            for hyp_path in hyp_paths:
                Path(hyp_path).unlink()

    @staticmethod
    def compute_subtoken_metrics(scores, hyp_paths, ref_path, lang1, lang2, data_set, deobfuscation_proba,
                                 is_teacher=False):
        prefix = f"{data_set}_{get_l1l2_string(lang1, lang2, deobfuscation_proba)}_mt_subtoken"
        if is_teacher and any(
                key.startswith(prefix) and
                key.endswith("_teacher") for
                key in scores.keys()):
            return
        subtoken_level_scores = run_subtoken_score(ref_path, hyp_paths)
        for score_type, value in subtoken_level_scores.items():
            logger.info(
                f"Subtoken {score_type} score {hyp_paths} {ref_path} {'teacher' if is_teacher else ''}: {value:f}"
            )
            scores[
                f"{prefix}_{score_type}" + ("_teacher" if is_teacher else "")
                ] = value

    def _generate_hypothesis(self, data_set, decoder, enc1, i, lang1, lang2, lang2_id, len1, params, x1, x2,
                             is_teacher=False):
        len_v = (3 * len1 + 10).clamp(max=params.max_len)
        enc1 = enc1.to(decoder.embeddings.weight.dtype)
        if params.beam_size == 1:
            if params.number_samples > 1:
                assert params.eval_temperature is not None
                generated, lengths = decoder.generate(
                    enc1.repeat_interleave(params.number_samples, dim=0),
                    len1.repeat_interleave(params.number_samples, dim=0),
                    lang2_id,
                    max_len=len_v.repeat_interleave(
                        params.number_samples, dim=0
                    ),
                    sample_temperature=params.eval_temperature,
                )
                generated = generated.T.reshape(
                    -1, params.number_samples, generated.shape[0]
                ).T
                lengths, _ = lengths.reshape(-1, params.number_samples).max(
                    dim=1
                )
            else:
                generated, lengths = decoder.generate(
                    enc1, len1, lang2_id, max_len=len_v
                )
            # print(f'path 1: {generated.shape}')

        else:
            assert params.number_samples == 1
            generated, lengths, _ = decoder.generate_beam(
                enc1,
                len1,
                lang2_id,
                beam_size=params.beam_size,
                length_penalty=params.length_penalty,
                early_stopping=params.early_stopping,
                max_len=len_v,
            )  # (len, beam, batch), (batch, beam)
            # print(f'path 2: {generated.shape}')
        if i == 0:
            # show 1 evaluation example and the corresponding model generation
            show_batch(
                logger,
                [
                    ("source", x1.transpose(0, 1)),
                    ("target", x2.transpose(0, 1)),
                    (
                        "gen",
                        generated.transpose(0, 1)
                        if len(generated.shape) == 2
                        else generated[:, 0, :].transpose(0, 1),
                    ),
                ],
                self.data["dico"],
                self.params.roberta_mode,
                f"{data_set} {lang1}-{lang2}{' teacher' if is_teacher else ''}",
            )
        return generated, lengths

    @staticmethod
    def _get_enc_dec(encoder, decoder, lang2_id, params):
        encoder = encoder[0].module if params.multi_gpu else encoder[0]
        decoder = (
            decoder[lang2_id] if params.separate_decoders else decoder[0]
        )
        decoder = decoder.module if params.multi_gpu else decoder
        return encoder, decoder

    @staticmethod
    def _model_forward(encoder, decoder, langs1, langs2, len1, len2, pred_mask, spans, x1, x2, y):
        # encode source sentence
        enc1 = encoder(
            "fwd", x=x1, lengths=len1, langs=langs1, causal=False, spans=spans
        )
        enc1 = enc1.transpose(0, 1)
        enc1 = enc1.to(decoder.embeddings.weight.dtype)

        # decode target sentence
        dec2 = decoder(
            "fwd",
            x=x2,
            lengths=len2,
            langs=langs2,
            causal=True,
            src_enc=enc1,
            src_len=len1,
            spans=spans,
        )
        # loss
        scores, loss = decoder(
            "predict", tensor=dec2, pred_mask=pred_mask, y=y, get_scores=True
        )
        return enc1, dec2, scores, loss

    @staticmethod
    def write_hypo_teacher_hypo_ref_src(
            data_set,
            hypothesis,
            teacher_hypothesis,
            lang1,
            lang2,
            params,
            references,
            scores,
            sources=None,
            deobfuscation_proba=None,
    ):
        # hypothesis paths
        t_hyp_paths = []
        # export sentences to hypothesis file / restore BPE segmentation
        for beam_number in range(len(teacher_hypothesis[0])):
            hyp_name = f"{get_l1l2_string(lang1, lang2, deobfuscation_proba)}.{data_set}_beam{beam_number}.teacher.txt"
            hyp_path = os.path.join(params.hyp_path, hyp_name)
            t_hyp_paths.append(hyp_path)
            print(f"outputting teacher hypotheses in {hyp_path}")
            with open(hyp_path, "w", encoding="utf-8") as f:
                f.write("\n".join([hyp[beam_number] for hyp in teacher_hypothesis]) + "\n")
            restore_segmentation(
                hyp_path, roberta_mode=params.roberta_mode, single_line=True
            )
        hyp_paths, ref_path, src_path = EncDecEvaluator.write_hypo_ref_src(data_set, hypothesis, lang1, lang2, params,
                                                                           references, scores, sources,
                                                                           deobfuscation_proba)
        return hyp_paths, t_hyp_paths, ref_path, src_path


def vizualize_do_files_student_teacher(lang1, src_file, ref_file, hyp_files, t_hyp_files):
    lang1_processor = LangProcessor.processors[lang1.split("_")[0]](
        root_folder=TREE_SITTER_ROOT
    )
    src_viz = str(Path(src_file).with_suffix(".vizualize.txt"))
    hyp_viz = str(
        Path(re.sub("beam\d", "", hyp_files[0])).with_suffix(".vizualize.txt.tmp")
    )
    t_hyp_viz = str(
        Path(re.sub("beam\d", "", t_hyp_files[0])).with_suffix(".vizualize.txt.tmp")
    )
    ref_viz = str(Path(ref_file).with_suffix(".vizualize.txt"))

    hyp_lines = list(
        zip(*[read_file_lines(path) for path in hyp_files])
    )  # test_size * beam_size
    t_hyp_lines = list(
        zip(*[read_file_lines(path) for path in t_hyp_files])
    )  # test_size * beam_size
    beam_size = len(hyp_lines[0])

    with open(src_file, encoding="utf-8") as f:
        src_lines = f.readlines()  # test_size

    with open(ref_file, encoding="utf-8") as f:
        ref_lines = f.readlines()  # test_size

    with open(src_viz, "w", encoding="utf-8") as src_vizf:
        with open(hyp_viz, "w", encoding="utf-8") as hyp_vizf:
            with open(t_hyp_viz, "w", encoding="utf-8") as t_hyp_vizf:
                with open(ref_viz, "w", encoding="utf-8") as ref_vizf:
                    src_vizf.write(
                        "========================SOURCE============================\n"
                    )
                    hyp_vizf.write(
                        "=====================STUDENT_HYPO=========================\n"
                    )
                    t_hyp_vizf.write(
                        "=====================TEACHER_HYPO=========================\n"
                    )
                    ref_vizf.write(
                        "==========================REF=============================\n"
                    )

                    for src, hyps, t_hyps, ref in zip(src_lines, hyp_lines, t_hyp_lines, ref_lines):
                        try:
                            src = lang1_processor.detokenize_code(src)
                            src_vizf.write(src)
                        except:
                            src = "".join(
                                [
                                    c if (i + 1) % 50 != 0 else c + "\n"
                                    for i, c in enumerate(src)
                                ]
                            )
                            src_vizf.write(src)

                        ref = ref.replace("|", "\n").strip()
                        ref_vizf.write(ref)

                        for i in range(beam_size):
                            hyp = hyps[i]
                            hyp = hyp.replace("|", "\n").strip()
                            hyp_vizf.write(hyp)
                            t_hyp = t_hyps[i]
                            t_hyp = t_hyp.replace("|", "\n").strip()
                            t_hyp_vizf.write(t_hyp)
                            if i == 0:
                                maximum = max(
                                    len(src.split("\n")),
                                    len(hyp.split("\n")),
                                    len(t_hyp.split("\n")),
                                    len(ref.split("\n")),
                                )
                                for i in range(len(src.split("\n")), maximum):
                                    src_vizf.write("\n")
                                for i in range(len(hyp.split("\n")), maximum):
                                    hyp_vizf.write("\n")
                                for i in range(len(t_hyp.split("\n")), maximum):
                                    t_hyp_vizf.write("\n")
                                for i in range(len(ref.split("\n")), maximum):
                                    ref_vizf.write("\n")
                            else:
                                maximum = max(
                                    len(src.split("\n")),
                                    len(hyp.split("\n")),
                                    len(t_hyp.split("\n")),
                                    len(ref.split("\n")),
                                )
                                for i in range(maximum - 1):
                                    src_vizf.write("\n")
                                for i in range(maximum - 1):
                                    ref_vizf.write("\n")
                                for i in range(len(hyp.split("\n")), maximum):
                                    hyp_vizf.write("\n")
                                for i in range(len(t_hyp.split("\n")), maximum):
                                    t_hyp_vizf.write("\n")
                            src_vizf.write("-\n")
                            hyp_vizf.write("-\n")
                            t_hyp_vizf.write("-\n")
                            ref_vizf.write("-\n")

                        src_vizf.write("--\n\n")
                        hyp_vizf.write("--\n\n")
                        t_hyp_vizf.write("--\n\n")
                        ref_vizf.write("--\n\n")

                        src_vizf.write(
                            "==========================================================\n"
                        )
                        hyp_vizf.write(
                            "==========================================================\n"
                        )
                        t_hyp_vizf.write(
                            "==========================================================\n"
                        )
                        ref_vizf.write(
                            "==========================================================\n"
                        )

    command = f"pr -w 320 -m -t {src_viz} {ref_viz} {hyp_viz} {t_hyp_viz} > {hyp_viz[:-4]}"
    subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).wait()

    os.remove(src_viz)
    os.remove(ref_viz)
    os.remove(hyp_viz)
    os.remove(t_hyp_viz)
