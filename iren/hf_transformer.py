import torch
from transformers import BertModel, BertConfig, GPT2Config, GPT2Model, PreTrainedModel

from codegen_sources.model.src.model import TransformerModel
from codegen_sources.model.src.model.transformer import N_MAX_POSITIONS, LAYER_NORM_EPSILON, PredLayer, \
    create_position_ids_from_input_ids, Embedding, create_sinusoidal_embeddings


class HFTransformer(TransformerModel):
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

        # transformer layers
        if self.is_encoder:
            self.encoder = BertModel(
                BertConfig(
                    vocab_size=len(dico),
                    hidden_size=self.dim,
                    num_hidden_layers=self.n_layers,
                    num_attention_heads=self.n_heads,
                    intermediate_size=self.hidden_dim,
                    hidden_act="gelu" if self.gelu_activation else "relu",
                    hidden_dropout_prob=self.dropout,
                    attention_probs_dropout_prob=self.attention_dropout,
                    max_position_embeddings=N_MAX_POSITIONS,
                    initializer_range=0.02,
                    layer_norm_eps=LAYER_NORM_EPSILON,
                    pad_token_id=self.pad_index,
                    position_embedding_type="absolute",
                    use_cache=True
                )
            )
        else:
            self.decoder = GPT2Model(
                GPT2Config(
                    is_decoder=True,
                    add_cross_attention=True,
                    vocab_size=len(dico),
                    n_positions=512,
                    n_embd=self.dim,
                    n_layer=self.n_layers,
                    n_head=self.n_heads,
                    n_inner=self.hidden_dim,
                    activation_function="gelu",
                    resid_pdrop=self.dropout,
                    embd_pdrop=self.dropout,
                    attn_pdrop=self.attention_dropout,
                    layer_norm_epsilon=LAYER_NORM_EPSILON,
                    initializer_range=0.02,
                    scale_attn_weights=True,
                    use_cache=True,
                    bos_token_id=self.eos_index,
                    eos_token_id=self.eos_index
                )
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

        # generate masks
        pad_mask = get_pad_mask(slen, lengths)

        # transformer layers
        if self.is_decoder and src_enc is not None and src_len is not None:
            src_mask = get_pad_mask(src_enc.shape[1], src_len)
            tensor = self.decoder(inputs_embeds=tensor,
                                  attention_mask=pad_mask,
                                  encoder_hidden_states=src_enc,
                                  encoder_attention_mask=src_mask)
            tensor = tensor
        else:
            tensor = self.encoder(inputs_embeds=tensor, attention_mask=pad_mask)

        # move back sequence length to dimension 0
        tensor = tensor.last_hidden_state.transpose(0, 1)

        return tensor


def get_pad_mask(slen, lengths):
    assert lengths.max().item() <= slen
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    pad_mask = alen < lengths[:, None]
    return pad_mask


def get_attn_mask(slen, lengths, causal):
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    return alen[None, :] > alen[:, None] if causal else None
