# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import argparse
import time
from pathlib import Path
import torch
from codegen_sources.model.src.logger import create_logger
from codegen_sources.model.src.utils import restore_roberta_segmentation_sentence
from codegen_sources.preprocessing.bpe_modes.fast_bpe_mode import FastBPEMode
from codegen_sources.preprocessing.bpe_modes.roberta_bpe_mode import RobertaBPEMode
from codegen_sources.preprocessing.lang_processors.lang_processor import LangProcessor
from codegen_sources.model.src.data.dictionary import (
    Dictionary,
    BOS_WORD,
    EOS_WORD,
    PAD_WORD,
    UNK_WORD,
    MASK_WORD,
)
from codegen_sources.model.src.model import build_model
from codegen_sources.model.src.utils import AttrDict
from iren.dataset_builder.source_dataset_mode import read_file
from iren.onnx import ONNXModel

SUPPORTED_LANGUAGES = ["java", "python"]

logger = create_logger(None, 0)


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Translate sentences")

    # model
    parser.add_argument("--model_path", type=str, default="", help="Model path")
    parser.add_argument(
        "--lang",
        type=str,
        default="",
        help=f"Code language, should be either {', '.join(SUPPORTED_LANGUAGES[:-1])} or {SUPPORTED_LANGUAGES[-1]}",
    )
    parser.add_argument(
        "--BPE_path",
        type=str,
        default=str(
            Path(__file__).parents[2].joinpath("data/bpe/cpp-java-python/codes")
        ),
        help="Path to BPE codes.",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Beam size. The beams will be printed in order of decreasing likelihood.",
    )

    return parser


class Deobfuscator:
    def __init__(self, model_path, BPE_path):
        self.reloaded_params, self.dico, (encoder, decoder) = _reload_model(model_path, gpu=False)
        self.encoder = encoder[0]
        self.decoder = decoder[0]
        print("Encoder:", self.encoder.parameters)
        print("Decoder:", self.decoder.parameters)
        self.encoder.eval()
        self.decoder.eval()

        # reload bpe
        if getattr(self.reloaded_params, "roberta_mode", False):
            self.bpe_model = RobertaBPEMode()
        else:
            self.bpe_model = FastBPEMode(
                codes=os.path.abspath(BPE_path), vocab_path=None
            )

    def deobfuscate(
            self, input, lang, n=1, beam_size=1, sample_temperature=None, device="cpu",
    ):

        # Build language processors
        assert lang, lang in SUPPORTED_LANGUAGES

        lang_processor = LangProcessor.processors[lang](
            root_folder=Path(__file__).parents[2].joinpath("tree-sitter")
        )
        obfuscator = lang_processor.obfuscate_code
        tokenizer = lang_processor.tokenize_code

        lang1 = lang + "_obfuscated"
        lang2 = lang + "_dictionary"
        lang1_id = self.reloaded_params.lang2id[lang1]
        lang2_id = self.reloaded_params.lang2id[lang2]

        assert (
                lang1 in self.reloaded_params.lang2id.keys()
        ), f"{lang1} should be in {self.reloaded_params.lang2id.keys()}"
        assert (
                lang2 in self.reloaded_params.lang2id.keys()
        ), f"{lang2} should be in {self.reloaded_params.lang2id.keys()}"

        print("Original Code:")
        print(input)

        dico = None
        # input, dico = obfuscator(input)
        # print("Obfuscated Code:")
        # print(input)

        with torch.no_grad():
            # Convert source code to ids
            start = time.perf_counter()
            tokens = [t for t in tokenizer(input)]
            print(f"Time: {time.perf_counter() - start}s")
            print(f"Tokenized {lang} function:")
            print(tokens)
            start = time.perf_counter()
            tokens = self.bpe_model.apply_bpe(" ".join(tokens))
            tokens = self.bpe_model.repair_bpe_for_obfuscation_line(tokens)
            print(f"Time: {time.perf_counter() - start}s")
            print(f"BPE {params.lang} function:")
            print(tokens)

            start = time.perf_counter()
            tokens = ["</s>"] + tokens.split() + ["</s>"]
            # tokens = tokens.split()
            input = " ".join(tokens)

            # Create torch batch
            len1 = len(input.split())
            len1 = torch.LongTensor(1).fill_(len1).to(device)
            x1 = torch.LongTensor([self.dico.index(w) for w in input.split()]).to(
                device
            )[:, None]
            langs1 = x1.clone().fill_(lang1_id)

            # ONNX inference (hardcoded)
            encoder = ONNXModel(self.dico,
                                "/home/igor/PycharmProjects/CodeGen/training_artifacts/onnx_models_old/encoder.opt.onnx")
            decoder = ONNXModel(self.dico,
                                "/home/igor/PycharmProjects/CodeGen/training_artifacts/onnx_models_old/decoder.opt.onnx")

            # x1 tensor([[1],
            #            [772],
            #            [581],
            #            [517],
            #            [2212],
            #            [517],
            #            [5057],
            #            [519],
            #            [553],
            #            [645],
            #            ...
            #            [519],
            #            [528],
            #            [528],
            #            [528],
            #            [1]])
            # len1 tensor([364])
            enc1 = encoder(x=x1, lengths=len1)
            # enc1 tensor([[[-1.2224, 0.4027, 0.4593, ..., 0.3197, 0.6488, 0.9330],
            #             [-1.2356, 0.4677, 0.4793, ..., 0.2686, 0.7487, 0.8674],
            #             [-1.3957, 0.4289, 0.4853, ..., 0.1434, 0.2772, -0.2484],
            #             ...,
            #             [-0.3108, 0.7761, 0.3697, ..., 1.2685, -1.2228, -0.1318],
            #             [-0.3755, 0.8178, 0.3500, ..., 1.3115, -1.1837, -0.2205],
            #             [-0.6207, 0.3150, 0.6154, ..., 0.3038, 0.4711, 1.0550]]])
            x2 = torch.ones((1, 1), dtype=torch.int64)
            # tensor([[1],
            #         [324],
            #         [1055]])
            len2 = torch.ones((1,), dtype=torch.int64)
            # tensor([3])
            langs2 = x2.clone().fill_(lang2_id)
            for _ in range(10):
                out = decoder(x=x2, lengths=len2, src_enc=enc1, src_len=len1)
                out_idx = out.argmax(1, keepdim=True)
                if out_idx == 1: break
                x2 = torch.cat((x2, out_idx), 0)
                langs2 = x2.clone().fill_(lang2_id)
                len2 += 1

            # encoder = EncoderToONNX(self.encoder)
            # decoder = DecoderToONNX(self.decoder)
            #
            # enc1 = encoder(x1, len1, langs=None)
            # x2 = torch.ones((1, 1), dtype=torch.int64)
            # len2 = torch.ones((1,), dtype=torch.int64)
            # langs2 = x2.clone().fill_(lang2_id)
            # for _ in range(10):
            #     out = decoder(x2, len2, langs=None, src_enc=enc1, src_len=len1)
            #     out_idx = out.argmax(1, keepdim=True)
            #     if out_idx == 1: break
            #     x2 = torch.cat((x2, out_idx), 0)
            #     langs2 = x2.clone().fill_(lang2_id)
            #     len2 += 1

            # # Encode
            # enc1 = self.encoder("fwd", x=x1, lengths=len1, langs=langs1, causal=False)
            # enc1 = enc1.transpose(0, 1)
            # if n > 1:
            #     enc1 = enc1.repeat(n, 1, 1)
            #     len1 = len1.expand(n)
            #
            # # Decode
            # if beam_size == 1:
            #     x2, len2 = self.decoder.generate(
            #         enc1,
            #         len1,
            #         lang2_id,
            #         max_len=int(
            #             min(self.reloaded_params.max_len, 3 * len1.max().item() + 10)
            #         ),
            #         sample_temperature=sample_temperature,
            #     )
            # else:
            #     x2, len2, _ = self.decoder.generate_beam(
            #         enc1,
            #         len1,
            #         lang2_id,
            #         max_len=int(
            #             min(self.reloaded_params.max_len, 3 * len1.max().item() + 10)
            #         ),
            #         early_stopping=True,
            #         length_penalty=1.0,
            #         beam_size=beam_size,
            #     )

            # Convert out ids to text
            tok = []
            for i in range(x2.shape[1]):
                wid = [self.dico[x2[j, i].item()] for j in range(len(x2))][1:]
                wid = wid[: wid.index(EOS_WORD)] if EOS_WORD in wid else wid
                if getattr(self.reloaded_params, "roberta_mode", False):
                    tok.append(restore_roberta_segmentation_sentence(" ".join(wid)))
                else:
                    tok.append(" ".join(wid).replace("@@ ", ""))
            results = []
            for t in tok:
                results.append(t)
            print(f"Time DOBF: {time.perf_counter() - start}s")
            return results, dico


def _reload_model(model_path, gpu=False):
    # reload model
    reloaded = torch.load(model_path, map_location="cpu")
    # change params of the reloaded model so that it will
    # relaod its own weights and not the MLM or DOBF pretrained model
    reloaded["params"]["reload_model"] = ",".join([model_path] * 2)
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

    # build model / reload weights (in the build_model method)
    return reloaded_params, dico, build_model(reloaded_params, dico, gpu)


if __name__ == "__main__":
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # check parameters
    assert os.path.isfile(
        params.model_path
    ), f"The path to the model checkpoint is incorrect: {params.model_path}"
    assert os.path.isfile(
        params.BPE_path
    ), f"The path to the BPE tokens is incorrect: {params.BPE_path}"
    assert (
            params.lang in SUPPORTED_LANGUAGES
    ), f"The source language should be in {SUPPORTED_LANGUAGES}."

    # Initialize translator
    deobfuscator = Deobfuscator(params.model_path, params.BPE_path)

    # read input code from stdin
    # input = read_file(f"/home/igor/PycharmProjects/CodeGen/examples/example{LANGUAGE_EXTENSIONS[params.lang]}")
    input = read_file(
        "/home/igor/IdeaProjects/spring-boot/spring-boot-project/spring-boot/src/main/java/org/springframework/boot/DefaultApplicationArguments.java")

    with torch.no_grad():
        output, dico = deobfuscator.deobfuscate(
            input, lang=params.lang, beam_size=params.beam_size,
        )

    print("True names:")
    print(dico)

    for out in output:
        print("=" * 20)
        print(out)
