import json
import logging
import time
import warnings
from collections import namedtuple
from typing import List

import torch
from flask import Flask, request

from codegen_sources.model.deobfuscate import Deobfuscator, SUPPORTED_LANGUAGES
from iren.inference.mute import mute_stderr
from codegen_sources.model.src.data.dictionary import EOS_WORD
from codegen_sources.model.src.model.transformer import N_MAX_POSITIONS
from codegen_sources.model.src.utils import restore_roberta_segmentation_sentence

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
# run_with_ngrok(app)
warnings.filterwarnings("ignore")

MODEL_PATH = "/models/DOBF_transcoder_size.pth"
BPE_PATH = "/data/bpe/cpp-java-python/codes"
LANGUAGE = "python"

VAR_TOKEN = "VAR_0"


class IdentifierPredictor(Deobfuscator):
    def predict(
            self, tokens, lang, n=1, beam_size=10, sample_temperature=None, device="cuda:0",
    ):

        # Build language processors
        assert lang, lang in SUPPORTED_LANGUAGES

        # lang_processor = LangProcessor.processors[lang](
        #     root_folder=Path(__file__).parents[2].joinpath("tree-sitter")
        # )
        # tokenizer = lang_processor.tokenize_code

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

        # print("Original Code:")
        # print(input)

        # input = obfuscator(input)[0]
        # print("Obfuscated Code:")
        # print(input)

        with torch.no_grad():
            # print(f"Tokenized {lang} function:")
            # print(tokens)
            with mute_stderr():
                tokens = self.bpe_model.apply_bpe(" ".join(tokens))
                tokens = self.bpe_model.repair_bpe_for_obfuscation_line(tokens)
            # print(f"BPE {lang} function:")
            # print(tokens)

            # limit number of bpe tokens
            start = max(0, tokens.index(VAR_TOKEN) - 256)  # magic number
            tokens = tokens[start: start + N_MAX_POSITIONS - 2]

            tokens = ["</s>"] + tokens.split() + ["</s>"]

            # Create torch batch
            len1 = len(tokens)
            len1 = torch.LongTensor(1).fill_(len1).to(device)
            x1 = torch.LongTensor([self.dico.index(w) for w in tokens]).to(
                device
            )[:, None]
            langs1 = x1.clone().fill_(lang1_id)

            # Encode
            enc1 = self.encoder("fwd", x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)
            if n > 1:
                enc1 = enc1.repeat(n, 1, 1)
                len1 = len1.expand(n)

            # Decode
            if beam_size == 1:
                x2, len2 = self.decoder.generate(
                    enc1,
                    len1,
                    lang2_id,
                    max_len=int(
                        min(self.reloaded_params.max_len, 3 * len1.max().item() + 10)
                    ),
                    sample_temperature=sample_temperature,
                )
            else:
                x2, len2, _ = self.decoder.generate_beam(
                    enc1,
                    len1,
                    lang2_id,
                    max_len=int(
                        min(self.reloaded_params.max_len, 3 * len1.max().item() + 10)
                    ),
                    early_stopping=True,
                    length_penalty=1.0,
                    beam_size=beam_size,
                )

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
            return results


predictor = IdentifierPredictor(MODEL_PATH, BPE_PATH)
LastInference = namedtuple("LastInference", ["predictions", "gt", "time_spent", "request_time"])
last_inference = LastInference(None, None, None, None)


@app.route('/', methods=['POST'])
def evaluate():
    if request.method == 'POST':
        start = time.perf_counter()
        variable_context = request.get_json(force=True, cache=False)
        context: List[str] = variable_context["context"]
        predictions = predictor.predict(context, LANGUAGE)
        # Remove "VAR_0 " from each prediction
        predictions = [prediction.replace("VAR_0 ", "") for prediction in predictions]
        time_spent = time.perf_counter() - start

        global last_inference
        last_inference = LastInference(predictions, variable_context["gt"], time_spent, time.asctime())
        return json.dumps({"predictions": predictions, "time": time_spent})


@app.route('/')
def running():
    global last_inference
    return "I don't have anything to show!" if last_inference.gt is None \
        else f"Running!<p>Last inference:<p>" + \
             f"Request time: {last_inference.request_time}<p>" + \
             f"Time spent: {last_inference.time_spent}<p>" + \
             f"<p>Ground truth: {last_inference.gt}<p>" + \
             f"Predictions: {last_inference.predictions}"


app.run()
