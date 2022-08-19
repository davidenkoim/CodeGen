import os

import torch
from transformers import EncoderDecoderConfig, EncoderDecoderModel, DistilBertConfig, GPT2Config, AutoTokenizer, \
    BertTokenizer, GPT2Tokenizer

from iren.hf_transformer import HFTransformer
from iren.onnx.convert import _reload_model, gen_random_batch

MODEL_PATH = "/home/igor/PycharmProjects/CodeGen/training_artifacts/models/distill_var_shuffled_F1_66.pth"
OUTPUT_DIR = "/home/igor/PycharmProjects/CodeGen/training_artifacts/onnx_models_hf"
OPSET_VERSION = 11
ENCODER_HF = os.path.join(OUTPUT_DIR, "encoder.hf")
DECODER_HF = os.path.join(OUTPUT_DIR, "decoder.hf")
ENCODER_ONNX = os.path.join(OUTPUT_DIR, "encoder.onnx")
DECODER_ONNX = os.path.join(OUTPUT_DIR, "decoder.onnx")

if __name__ == "__main__":
    params, dico, (encoder, decoder) = _reload_model(MODEL_PATH)
    encoder = HFTransformer(params, dico, is_encoder=True)
    decoder = HFTransformer(params, dico, is_encoder=False)

    x1, len1, langs1, x2, len2, langs2 = gen_random_batch()

    enc1 = encoder(x1, len1)

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

    os.system(f"python -m onnxruntime.transformers.optimizer -m {ENCODER_HF} --opset {OPSET_VERSION} {ENCODER_ONNX}")
    os.system(f"python -m transformers.onnx -m {ENCODER_HF} --opset {OPSET_VERSION} {DECODER_ONNX}")
