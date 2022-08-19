from logging import getLogger
from pathlib import Path

import numpy as np
import torch
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
from tqdm import tqdm

from codegen_sources.model.src.evaluation.evaluator import EVAL_DATASET_SPLITS, get_l1l2_string, \
    SRC_ST_LANGS, TARGET_ST_LANG, EncDecEvaluator, eval_moses_bleu
from codegen_sources.model.src.evaluation.subtoken_score import run_subtoken_score
from codegen_sources.model.src.utils import add_noise, convert_to_text, get_programming_language_name, \
    vizualize_do_files, show_batch
from iren.onnx import to_numpy
from iren.utils.obfuscation import deobfuscate_by_variable
from iren.utils.ranking_metrics import run_ranking_scoring

logger = getLogger()


class ONNXEvaluator(EncDecEvaluator):
    def __init__(self, encoder, decoder, data, params):
        super(EncDecEvaluator, self).__init__(None, data, params)
        self.encoder = encoder
        self.decoder = decoder

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

        # store hypothesis to compute BLEU score
        if params.eval_bleu_test_only:
            datasets_for_bleu = ["test"]
        else:
            datasets_for_bleu = [s for s in EVAL_DATASET_SPLITS if s != "train"]

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        for deobfuscation_proba in (
                deobfuscate_probas if deobfuscate_probas is not None else [None]
        ):
            if deobfuscate:
                rng = np.random.RandomState(0)
            l1l2 = get_l1l2_string(lang1, lang2, deobfuscation_proba)
            eval_name = f"{data_set}_{l1l2}"

            hypothesis = []
            sources = []
            references = []
            for i, batch in enumerate(
                    tqdm(self.get_iterator(
                        data_set, lang1, lang2 if lang2 != lang1 else None, span=span
                    ), desc=f"local-rank-{params.local_rank}-obf_proba-{deobfuscation_proba}", mininterval=1.)
            ):
                if i >= params.limit_test_size:
                    break
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
                            (x1, len1, x2, len2) = deobfuscate_by_variable(
                                x1, x2, deobfuscation_proba, params.roberta_mode,
                                self.data["dico"], params.pad_index, params.eos_index, params.max_len,
                                rng=rng,
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

                try:
                    enc1 = self.encoder(x=x1, lengths=len1, langs=langs1)
                except InvalidArgument:
                    enc1 = self.encoder(x=x1, lengths=len1)
                # generate translation - translate / convert to text
                if (
                        eval_bleu or eval_computation or eval_subtoken_score
                ) and data_set in datasets_for_bleu:
                    generated, lengths = self._generate_hypothesis(data_set, self.decoder, enc1, i, lang1, lang2,
                                                                   lang2_id, len1, params, x1, x2)

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

            # write hypotheses
            if (
                    eval_bleu or eval_computation or eval_subtoken_score
            ) and data_set in datasets_for_bleu:
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

            # compute BLEU score
            if eval_bleu and data_set in datasets_for_bleu:
                self.evaluate_bleu(scores, hyp_paths, ref_path, deobfuscation_proba, lang1, lang2, eval_computation,
                                   data_set)

            if params.eval_ranking_metrics and params.beam_size > 1:
                self.compute_ranking_metrics(scores, hyp_paths, ref_path, deobfuscation_proba, lang1, lang2, data_set)

            if (
                    deobfuscate
                    and eval_bleu
                    or eval_subtoken_score
                    and data_set in datasets_for_bleu
            ):
                # TODO clean lang1
                vizualize_do_files(lang1.split("_")[0], src_path, ref_path, hyp_paths)

    @staticmethod
    def compute_ranking_metrics(scores, hyp_paths, ref_path, deobfuscation_proba, lang1, lang2, data_set):
        prefix = f"{data_set}_{get_l1l2_string(lang1, lang2, deobfuscation_proba)}_mt_ranking"
        ranking_metrics = run_ranking_scoring(ref_path, hyp_paths)
        for metric_type, value in ranking_metrics.items():
            logger.info(f"Ranking {metric_type} score {hyp_paths} {ref_path}: {value:f}")
            scores[f"{prefix}_{metric_type}"] = value

    @staticmethod
    def evaluate_bleu(scores, hyp_paths, ref_path, deobfuscation_proba, lang1, lang2, eval_computation, data_set):
        prefix = f"{data_set}_{get_l1l2_string(lang1, lang2, deobfuscation_proba)}_mt_bleu"
        # evaluate BLEU score
        bleu = eval_moses_bleu(ref_path, hyp_paths[0])
        logger.info(f"BLEU {hyp_paths[0]} {ref_path}: {bleu:f}")
        scores[prefix] = bleu
        if eval_computation:
            for hyp_path in hyp_paths:
                Path(hyp_path).unlink()

    @staticmethod
    def compute_subtoken_metrics(scores, hyp_paths, ref_path, lang1, lang2, data_set, deobfuscation_proba):
        prefix = f"{data_set}_{get_l1l2_string(lang1, lang2, deobfuscation_proba)}_mt_subtoken"
        subtoken_level_scores = run_subtoken_score(ref_path, hyp_paths)
        for score_type, value in subtoken_level_scores.items():
            logger.info(f"Subtoken {score_type} score {hyp_paths} {ref_path}: {value:f}")
            scores[f"{prefix}_{score_type}"] = value

    def _generate_hypothesis(self, data_set, decoder, enc1, i, lang1, lang2, lang2_id, len1, params, x1, x2,
                             is_teacher=False):
        len_v = (3 * len1 + 10).clamp(max=params.max_len)
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
