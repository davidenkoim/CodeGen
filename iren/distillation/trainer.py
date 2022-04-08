import sys
import time
from logging import getLogger
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from codegen_sources.model.src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from codegen_sources.model.src.model import build_model
from codegen_sources.model.src.trainer import EncDecTrainer
from codegen_sources.model.src.utils import (
    add_noise,
    AttrDict,
)
from codegen_sources.model.src.utils import to_cuda, show_batch

sys.path.append(str(Path(__file__).parents[3]))
print("adding to path", str(Path(__file__).parents[3]))

logger = getLogger()


class DistillationTrainer(EncDecTrainer):
    def __init__(self, encoder, decoder, data, params):
        super().__init__(encoder, decoder, data, params)
        logger.info("Using DistillationTrainer...")
        # reload model
        reloaded = torch.load(params.teacher_path, map_location="cpu")
        # change params of the reloaded model so that it will
        # reload its own weights and not the MLM or DOBF pretrained model
        reloaded["params"]["reload_model"] = ",".join([params.teacher_path] * 2)
        reloaded["params"]["lgs_mapping"] = ""
        reloaded["params"]["reload_encoder_for_decoder"] = False
        reloaded_params = AttrDict(reloaded["params"])

        # build dictionary / update parameters
        dico = Dictionary(
            reloaded["dico_id2word"], reloaded["dico_word2id"], reloaded["dico_counts"]
        )
        assert reloaded_params.n_words == len(dico)
        assert reloaded_params.bos_index == dico.index(BOS_WORD)
        assert reloaded_params.eos_index == dico.index(EOS_WORD)
        assert reloaded_params.pad_index == dico.index(PAD_WORD)
        assert reloaded_params.unk_index == dico.index(UNK_WORD)
        assert reloaded_params.mask_index == dico.index(MASK_WORD)

        assert data["dico"].word2id == dico.word2id

        # build model / reload weights (in the build_model method)
        self.teacher_encoder, self.teacher_decoder = build_model(reloaded_params, dico)
        self.teacher_encoder = [model.cuda().eval() for model in self.teacher_encoder]
        self.teacher_decoder = [model.cuda().eval() for model in self.teacher_decoder]

        self.kld_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.temperature_kld = params.temperature_kld
        self.lambda_kld = params.lambda_kld

    def mt_step(
            self,
            lang1,
            lang2,
            lambda_coeff,
            span=None,
            deobfuscate=False,
            deobfuscate_p=None,
            show_example=False,
    ):
        """
        Machine translation step.
        Can also be used for denoising auto-encoding.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        assert (
                       deobfuscate_p is not None and 0 <= deobfuscate_p <= 1
               ) or not deobfuscate
        # assert deobfuscate or span is not None
        params = self.params
        self.train_mode()

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        spans = None
        # generate batch
        if lang1 == lang2:
            assert not span, "spans not supported for AE steps"
            (x1, len1, _, _) = self.get_batch("ae", lang1)
            (x2, len2) = (x1, len1)
            (x1, len1) = add_noise(x1, len1, self.params, len(self.data["dico"]) - 1)
        elif span:
            (
                (x1, len1, _, _),
                (x2, len2, _, _),
                (spans, len_spans, _, _),
            ) = self.get_batch("mt_spans", lang1, lang2, span=span)
        elif deobfuscate:
            (x1, len1, _, _), (x2, len2, _, _) = self.get_batch("mt", lang1, lang2)
            (x1, len1, x2, len2) = self.deobfuscate_by_variable(
                x1, x2, deobfuscate_p, params.roberta_mode, rng=None
            )
            if x1 is None:
                return
        else:
            (x1, len1, _, _), (x2, len2, _, _) = self.get_batch("mt", lang1, lang2)

        # log first batch of training
        if show_example:
            show_batch(
                logger,
                [("source", x1.transpose(0, 1)), ("target", x2.transpose(0, 1))],
                self.data["dico"],
                self.params.roberta_mode,
                f"Train {lang1}-{lang2}",
            )

        langs1 = x1.clone().fill_(lang1_id)
        langs2 = x2.clone().fill_(lang2_id)

        # target words to predict
        alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
        # do not predict anything given the last target word
        pred_mask = alen[:, None] < len2[None] - 1
        y = x2[1:].masked_select(pred_mask[:-1])
        assert len(y) == (len2 - 1).sum().item()

        # cuda
        x1, len1, langs1, x2, len2, langs2, y, spans = to_cuda(
            x1, len1, langs1, x2, len2, langs2, y, spans
        )

        student_encoder, student_decoder = self._get_enc_dec(self.encoder, self.decoder, lang2_id, params)
        _, _, scores, loss = self.model_forward(student_encoder, student_decoder,
                                                langs1, langs2, len1, len2, pred_mask, spans, x1, x2, y)

        teacher_encoder, teacher_decoder = self._get_enc_dec(self.teacher_encoder, self.teacher_decoder, lang2_id,
                                                             params)
        with torch.no_grad():
            _, _, t_scores, t_loss = self.model_forward(teacher_encoder, teacher_decoder,
                                                        langs1, langs2, len1, len2, pred_mask, spans, x1, x2, y)

        kld_loss = self.kld_loss_fct(F.log_softmax(scores / self.temperature_kld, dim=-1),
                                     F.softmax(t_scores / self.temperature_kld, -1)) * self.temperature_kld ** 2

        if deobfuscate:
            self.stats_append("DO-%s-%s" % (lang1, lang2), loss.item())
            self.stats_append("DO-t-%s-%s" % (lang1, lang2), t_loss.item())
        else:
            key = (lang1, lang2) if span is None else (lang1, lang2, span)
            self.stats_append(("AE-%s" % lang1) if lang1 == lang2 else ("MT-%s" % "-".join(key)), loss.item())
            self.stats_append(("AE-t-%s" % lang1) if lang1 == lang2 else ("MT-t-%s" % "-".join(key)), t_loss.item())

        self.stats_append(f"KLD-{lang1}-{lang2}", kld_loss.item())

        n_words = y.size(0)
        n_valid = (scores.max(1)[1] == y).sum().item()
        t_n_valid = (t_scores.max(1)[1] == y).sum().item()
        self.stats_append(f"BPE-acc-{lang1}-{lang2}", n_valid / n_words)
        self.stats_append(f"BPE-acc-t-{lang1}-{lang2}", t_n_valid / n_words)

        loss_sum = lambda_coeff * loss + self.lambda_kld * kld_loss

        # optimize
        self.optimize(loss_sum)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats["processed_s"] += len2.size(0)
        self.stats["processed_w"] += (len2 - 1).sum().item()

    @staticmethod
    def _get_enc_dec(encoder, decoder, lang2_id, params):
        enc = encoder[0]
        dec = decoder[lang2_id] if params.separate_decoders else decoder[0]
        return enc, dec

    @staticmethod
    def model_forward(encoder, decoder, langs1, langs2, len1, len2, pred_mask, spans, x1, x2, y):
        # encode source sentence
        enc1 = encoder(
            "fwd", x=x1, lengths=len1, langs=langs1, causal=False, spans=spans
        )
        enc1 = enc1.transpose(0, 1)
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

    def stats_append(self, key, value):
        if key not in self.stats:
            self.stats[key] = []
        self.stats[key].append(value)

    def print_stats(self):
        """
        Print statistics about the training.
        """
        # if self.n_total_iter % 5 != 0:
        #     return

        s_iter = "%7i - " % self.n_total_iter
        d_stats = {k: np.mean(v)
                   for k, v in self.stats.items()
                   if isinstance(v, list) and len(v) > 0}
        s_stat = "\n".join(
            [
                "{}:\t{:7.4f}".format(k, np.mean(v))
                for k, v in d_stats.items()
            ]
        )
        for k, v in self.stats.items():
            if type(v) is list:
                del v[:]

        # learning rates
        s_lr = ""
        for k, v in self.optimizers.items():
            s_lr = (
                    s_lr
                    + (" - %s LR: " % k)
                    + " / ".join("{:.4e}".format(group["lr"]) for group in v.param_groups)
            )

        if self.params.bt_sample_temperature > 0:
            s_bt_samp = " - BT-sampling-T: " + "{:2.2e}".format(
                self.params.bt_sample_temperature
            )
        else:
            s_bt_samp = ""

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} sent/s - {:8.2f} words/s".format(
            self.stats["processed_s"] * 1.0 / diff,
            self.stats["processed_w"] * 1.0 / diff,
        )
        self.stats["processed_s"] = 0
        self.stats["processed_w"] = 0
        self.last_time = new_time

        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_lr + s_bt_samp + "\n" + s_stat)
