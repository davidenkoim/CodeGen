import torch
import torch.nn.functional as F
from torch import nn

from codegen_sources.model.src.model import TransformerModel
from codegen_sources.model.src.model.transformer import Embedding, N_MAX_POSITIONS, create_sinusoidal_embeddings, \
    LAYER_NORM_EPSILON, PredLayer, create_position_ids_from_input_ids


class Transformer(TransformerModel):
    def __init__(self, params, dico, is_encoder, with_output):
        """
        Transformer model (encoder or decoder).
        """
        super(TransformerModel, self).__init__()

        # encoder / decoder, output layer
        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder
        self.with_output = with_output
        self.use_span_embeddings = params.spans_emb_encoder and self.is_encoder

        # dictionary / languages
        self.n_langs = params.n_langs
        self.n_words = params.n_words
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.dico = dico
        self.id2lang = params.id2lang
        self.lang2id = params.lang2id
        self.use_lang_emb = getattr(params, "use_lang_emb", True)
        assert len(self.dico) == self.n_words
        assert len(self.id2lang) == len(self.lang2id) == self.n_langs

        # model parameters
        self.dim = (
            params.emb_dim_encoder if is_encoder else params.emb_dim_decoder
        )  # 512 by default
        self.hidden_dim = self.dim * 4  # 2048 by default
        self.n_heads = params.n_heads  # 8 by default
        self.n_layers = (
            params.n_layers_encoder if is_encoder else params.n_layers_decoder
        )
        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        self.roberta_mode = getattr(params, "roberta_mode", False)
        self.gelu_activation = params.gelu_activation
        assert self.gelu_activation or not self.roberta_mode
        # assert self.dim % self.n_heads == 0, 'transformer dim must be a multiple of n_heads'

        # embeddings
        if self.roberta_mode:
            self.position_embeddings = Embedding(
                N_MAX_POSITIONS, self.dim, self.pad_index
            )
        else:
            self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
        if params.sinusoidal_embeddings:
            create_sinusoidal_embeddings(
                N_MAX_POSITIONS, self.dim, out=self.position_embeddings.weight
            )
        if params.n_langs > 0 and self.use_lang_emb:
            self.lang_embeddings = Embedding(self.n_langs, self.dim)
        self.embeddings = Embedding(self.n_words, self.dim, padding_idx=self.pad_index)
        if self.use_span_embeddings:
            # self.spans_embeddings = Embedding(self.n_words, self.dim, padding_idx=self.pad_index)
            self.spans_embeddings = Embedding(
                params.n_classes_classif, self.dim, padding_idx=self.pad_index
            )
        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=LAYER_NORM_EPSILON)

        # transformer layers
        if self.is_encoder:
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.dim,
                    nhead=self.n_heads,
                    dim_feedforward=self.hidden_dim,
                    dropout=self.attention_dropout,
                    activation="gelu" if self.gelu_activation else "relu",
                    layer_norm_eps=LAYER_NORM_EPSILON,
                    batch_first=True,
                    norm_first=True
                ),
                self.n_layers,
                norm=nn.LayerNorm(self.dim, eps=LAYER_NORM_EPSILON)
            )
        else:
            self.decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=self.dim,
                    nhead=self.n_heads,
                    dim_feedforward=self.hidden_dim,
                    dropout=self.attention_dropout,
                    activation="gelu" if self.gelu_activation else "relu",
                    layer_norm_eps=LAYER_NORM_EPSILON,
                    batch_first=True,
                    norm_first=True
                ),
                self.n_layers,
                norm=nn.LayerNorm(self.dim, eps=LAYER_NORM_EPSILON)
            )

        # output layer
        if self.with_output:
            self.pred_layer = PredLayer(params)
            if params.share_inout_emb:
                self.pred_layer.proj.weight = self.embeddings.weight

    def fwd(
            self,
            x,
            lengths,
            causal,
            src_enc=None,
            src_len=None,
            positions=None,
            langs=None,
            use_cache=False,
            spans=None,
    ):
        """
        Inputs:
            `x` LongTensor(slen, bs), containing word indices
            `lengths` LongTensor(bs), containing the length of each sentence
            `causal` Boolean, if True, the attention is only done over previous hidden states
            `positions` LongTensor(slen, bs), containing word positions
            `langs` LongTensor(slen, bs), containing language IDs
            `spans` LongTensor(slen, bs), containing the spans if use_spans is set to True
        """
        # lengths = (x != self.pad_index).float().sum(dim=1)
        # mask = x != self.pad_index

        assert not (use_cache and self.cache is None)
        if self.use_span_embeddings:
            assert spans is not None
        # check inputs
        slen, bs = x.size()
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        x = x.transpose(0, 1)  # batch size as dimension 0
        assert (src_enc is None) == (src_len is None)
        if src_enc is not None:
            assert self.is_decoder
            assert src_enc.size(0) == bs

        # positions
        if positions is None:
            if self.roberta_mode:
                positions = create_position_ids_from_input_ids(x, self.pad_index)
            else:
                positions = torch.arange(slen, dtype=torch.long, device=x.device).unsqueeze(0)
        else:
            assert positions.size() == (slen, bs)
            positions = positions.transpose(0, 1)

        # langs
        if langs is not None:
            assert langs.size() == (slen, bs)
            langs = langs.transpose(0, 1)

        # embeddings
        tensor = self.embeddings(x)
        if self.use_span_embeddings:
            tensor = tensor + self.spans_embeddings(spans)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        if langs is not None and self.use_lang_emb:
            tensor = tensor + self.lang_embeddings(langs)
        tensor = self.layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        # tensor *= ~pad_mask.unsqueeze(-1).to(tensor.dtype)

        # generate masks
        pad_mask = get_pad_mask(slen, lengths)
        attn_mask = get_attn_mask(slen, lengths, causal)

        # transformer layers
        if self.is_decoder and src_enc is not None and src_len is not None:
            src_mask = get_pad_mask(src_enc.shape[1], src_len)
            tensor = self.decoder(tgt=tensor,
                                  memory=src_enc,
                                  tgt_mask=attn_mask,
                                  memory_mask=None,
                                  tgt_key_padding_mask=pad_mask,
                                  memory_key_padding_mask=src_mask)
        else:
            tensor = self.encoder(src=tensor, mask=attn_mask, src_key_padding_mask=pad_mask)

        # move back sequence length to dimension 0
        tensor = tensor.transpose(0, 1)

        return tensor


def get_pad_mask(slen, lengths):
    assert lengths.max().item() <= slen
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    pad_mask = alen >= lengths[:, None]
    return pad_mask


def get_attn_mask(slen, lengths, causal):
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    return alen[None, :] > alen[:, None] if causal else None
