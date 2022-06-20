import os.path

import numpy as np
import onnxruntime as rt
import torch
from onnxruntime.quantization import quantize_dynamic
from torch import nn

from codegen_sources.model.src.data.dictionary import Dictionary
from codegen_sources.model.src.model import build_model, TransformerModel
from codegen_sources.model.src.utils import AttrDict
# TODO: use hydra
from iren.hf_transformer import HFTransformer
from iren.onnx import to_numpy
from iren.transformer import Transformer

MODEL_PATH = "/home/igor/PycharmProjects/CodeGen/training_artifacts/models/distill_var_shuffled_F1_66.pth"
OLD = False
OUTPUT_DIR = "/home/igor/PycharmProjects/CodeGen/training_artifacts/onnx_models"
OPSET_VERSION = 13


def gen_random_batch(batch_size=5):
    x1 = torch.randint(64000, size=(100, batch_size))
    len1 = torch.randint(x1.size(0), size=(batch_size,))
    langs1 = torch.ones_like(x1)
    x2 = torch.randint(64000, size=(10, batch_size))
    len2 = torch.randint(x2.size(0), size=(batch_size,))
    langs2 = torch.ones_like(x2)
    return x1, len1, langs1, x2, len2, langs2


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

    # build model / reload weights (in the build_model method)
    return reloaded_params, dico, build_model(reloaded_params, dico, gpu)


@torch.no_grad()
def time_forward(encoder, decoder, langs1, langs2, len1, len2, pred_mask, spans, x1, x2, y):
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
    scores = decoder.pred_layer.get_scores(dec2[-1, :, :])


class EncoderToONNX(nn.Module):
    def __init__(self, encoder: TransformerModel):
        super().__init__()
        self.encoder = encoder

    def forward(self, x, lengths, langs=None):
        out = self.encoder('fwd', x=x, lengths=lengths, langs=langs, causal=False)
        return out.transpose(0, 1)


class DecoderToONNX(nn.Module):
    def __init__(self, decoder: TransformerModel):
        super().__init__()
        self.decoder = decoder

    def forward(self, x, lengths, langs=None, src_enc=None, src_len=None):
        """
        :return: the last logit for each sentence.
        """
        dec2 = self.decoder(
            "fwd",
            x=x,
            lengths=lengths,
            langs=langs,
            causal=True,
            src_enc=src_enc,
            src_len=src_len,
            spans=None,
        )
        # dec_flat = dec2.reshape(-1, dec2.size(-1))
        # batch_size = lengths.size(0)
        # idxs = (lengths - 1) * batch_size + torch.arange(batch_size)
        return self.decoder.pred_layer.get_scores(dec2[-1, :, :])  # (B, V)


def save_dico(dico):
    dico_path = os.path.join(OUTPUT_DIR, "vocab.txt")
    with open(dico_path, "w") as f:
        f.writelines(f"{word}\n" for _, word in sorted(dico.id2word.items()))


if __name__ == "__main__":
    params, dico, (encoder, decoder) = _reload_model(MODEL_PATH)
    if OLD:
        encoder = encoder[0]
        decoder = decoder[0]
    else:
        encoder = HFTransformer(params, dico, is_encoder=True, with_output=False)
        decoder = HFTransformer(params, dico, is_encoder=False, with_output=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_dico(dico)
    encoder = EncoderToONNX(encoder.eval())
    decoder = DecoderToONNX(decoder.eval())
    x1, len1, langs1, x2, len2, langs2 = gen_random_batch()
    if not params.use_lang_emb:
        langs1, langs2 = None, None

    enc1 = encoder(x1, len1, langs=langs1)

    encoder_path = os.path.join(OUTPUT_DIR, "encoder.onnx")
    encoder_opt_path = os.path.join(OUTPUT_DIR, "encoder.opt.onnx")
    encoder_quant_path = os.path.join(OUTPUT_DIR, "encoder.quant.onnx")
    # without langs embeddings for now
    torch.onnx.export(encoder,
                      (x1, len1),
                      encoder_path,
                      export_params=True,
                      do_constant_folding=True,
                      verbose=True,
                      opset_version=OPSET_VERSION,
                      input_names=['x', 'lengths'],
                      output_names=['output'],
                      dynamic_axes=dict(x={0: 'seq_length', 1: 'batch_size'}, lengths={0: 'batch_size'},
                                        output={0: 'batch_size'}))
    sess_options = rt.SessionOptions()

    # Set graph optimization level
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    # To enable model serialization after graph optimization set this
    sess_options.optimized_model_filepath = encoder_opt_path

    session = rt.InferenceSession(encoder_path, sess_options)

    quantize_dynamic(encoder_opt_path, encoder_quant_path)

    ort_encoder_session = rt.InferenceSession(encoder_opt_path)
    ort_encoder_inputs = dict(x=to_numpy(x1), lengths=to_numpy(len1))
    ort_encoder_outputs = ort_encoder_session.run(['output'], ort_encoder_inputs)
    print("Encoder output:\n", to_numpy(enc1), "\n", ort_encoder_outputs[0])
    try:
        np.testing.assert_allclose(ort_encoder_outputs[0], to_numpy(enc1), rtol=1e-3, atol=1e-5)
    except Exception as e:
        print(e)

    dec2 = decoder(x2, len2, langs=langs2, src_enc=enc1, src_len=len1)

    decoder_path = os.path.join(OUTPUT_DIR, "decoder.onnx")
    decoder_opt_path = os.path.join(OUTPUT_DIR, "decoder.opt.onnx")
    decoder_quant_path = os.path.join(OUTPUT_DIR, "decoder.quant.onnx")
    # without langs embeddings for now
    torch.onnx.export(decoder,
                      (x2, len2, {'src_enc': enc1, 'src_len': len1}),
                      decoder_path,
                      export_params=True,
                      do_constant_folding=True,
                      verbose=True,
                      opset_version=OPSET_VERSION,
                      input_names=['x', 'lengths', 'src_enc', 'src_len'],
                      output_names=['output'],
                      dynamic_axes=dict(x={0: 'seq_length', 1: 'batch_size'}, lengths={0: 'batch_size'},
                                        src_enc={0: 'batch_size', 1: 'enc_seq_length'}, src_len={0: 'batch_size'},
                                        output={0: 'batch_size'}))
    sess_options = rt.SessionOptions()

    # Set graph optimization level
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    # To enable model serialization after graph optimization set this
    sess_options.optimized_model_filepath = decoder_opt_path

    session = rt.InferenceSession(decoder_path, sess_options)

    quantize_dynamic(decoder_opt_path, decoder_quant_path)

    ort_decoder_session = rt.InferenceSession(decoder_opt_path)
    ort_decoder_inputs = dict(x=to_numpy(x2),
                              lengths=to_numpy(len2),
                              src_enc=ort_encoder_outputs[0],
                              src_len=to_numpy(len1))
    ort_decoder_outputs = ort_decoder_session.run(['output'], ort_decoder_inputs)
    print("Decoder output:\n", to_numpy(dec2), "\n", ort_decoder_outputs[0])
    try:
        np.testing.assert_allclose(ort_decoder_outputs[0], to_numpy(dec2), rtol=1e-3, atol=1e-5)
    except Exception as e:
        print(e)

"""
Traceback (most recent call last):
  File "/home/igor/anaconda3/envs/codeGen_env/lib/python3.7/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 335, in __init__
    self._create_inference_session(providers, provider_options, disabled_optimizers)
  File "/home/igor/anaconda3/envs/codeGen_env/lib/python3.7/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 370, in _create_inference_session
    sess = C.InferenceSession(session_options, self._model_path, True, self._read_config_from_model)
onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL : Load model from /home/igor/PycharmProjects/CodeGen/training_artifacts/onnx_models/decoder.quant.onnx failed:Node (Add_47) Op (Add) [ShapeInferenceError] Incompatible dimensions
"""
