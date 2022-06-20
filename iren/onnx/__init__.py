import onnxruntime as rt
import torch
import torch.nn.functional as F
from numpy import ndarray
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

from codegen_sources.model.src.model.transformer import BeamHypotheses


def to_numpy(tensor):
    if isinstance(tensor, ndarray):
        return tensor
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class ONNXModel:
    """
    Wrapper model of onnxruntime.InferenceSession
    """

    def __init__(self, dico, path):
        self.n_words = len(dico)
        self.pad_index = dico.pad_index
        self.eos_index = dico.eos_index
        self.session = rt.InferenceSession(path)

    def __call__(self, **kargs):
        inputs = {k: to_numpy(v) for k, v in kargs.items()}
        return torch.tensor(self.session.run(['output'], inputs)[0])

    def generate(
            self, src_enc, src_len, tgt_lang_id, max_len=200, sample_temperature=None
    ):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        `langs`:
            - must be None if the model only supports one language
            - lang_id if only one language is involved (LM)
            - (lang_id1, lang_id2) if two languages are involved (MT)
        """

        if isinstance(max_len, int):
            max_lengths = src_len.clone().fill_(max_len)
            global_max_len = max_len
        else:
            max_lengths = max_len
            global_max_len = int(max_lengths.max())

        # input batch
        bs = len(src_len)
        assert src_enc.size(0) == bs

        # generated sentences
        generated = src_len.new(global_max_len, bs)  # upcoming output
        generated.fill_(self.pad_index)  # fill upcoming ouput with <PAD>
        generated[0].fill_(self.eos_index)  # we use <EOS> for <BOS> everywhere

        # positions
        positions = src_len.new(global_max_len).long()
        positions = (
            torch.arange(global_max_len, out=positions)
            .unsqueeze(1)
            .expand(global_max_len, bs)
        )

        # language IDs
        langs = src_len.new(global_max_len).long().fill_(tgt_lang_id)
        langs = langs.unsqueeze(1).expand(global_max_len, bs)

        # current position / max lengths / length of generated sentences / unfinished sentences
        cur_len = 1
        gen_len = src_len.clone().fill_(1)
        unfinished_sents = src_len.clone().fill_(1)

        # cache compute states
        self.cache = {"slen": 0}
        previous_unfinished_mask = unfinished_sents.ne(0)
        while cur_len < global_max_len:
            # compute word scores
            unfinished_mask = unfinished_sents.ne(0)

            should_modify = unfinished_mask.ne(previous_unfinished_mask).any()
            restricted_mask = unfinished_mask[previous_unfinished_mask]

            if should_modify and self.cache is not None:
                for k, v in self.cache.items():
                    if isinstance(k, int):
                        assert len(v) == 2
                        self.cache[k] = (
                            cached_tensor[restricted_mask] for cached_tensor in v
                        )

            # TODO
            # tensor = self.forward(
            #     "fwd",
            #     x=generated[:cur_len, unfinished_mask],
            #     lengths=gen_len[unfinished_mask],
            #     positions=positions[:cur_len, unfinished_mask],
            #     langs=langs[:cur_len][:, unfinished_mask],
            #     causal=True,
            #     src_enc=src_enc[unfinished_mask],
            #     src_len=src_len[unfinished_mask],
            #     use_cache=True,
            # )
            # tensor = tensor.data[-1, :, :].type_as(src_enc)  # (bs, dim)
            # scores = self.pred_layer.get_scores(tensor)  # (bs, n_words)
            # ---
            try:
                scores = self(
                    x=generated[:cur_len, unfinished_mask],
                    lengths=gen_len[unfinished_mask],
                    langs=langs[:cur_len][:, unfinished_mask],
                    src_enc=src_enc[unfinished_mask],
                    src_len=src_len[unfinished_mask]
                )
            except InvalidArgument:
                scores = self(
                    x=generated[:cur_len, unfinished_mask],
                    lengths=gen_len[unfinished_mask],
                    src_enc=src_enc[unfinished_mask],
                    src_len=src_len[unfinished_mask]
                )
            # TODO end

            # select next words: sample or greedy
            if sample_temperature is None:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            else:
                next_words = torch.multinomial(
                    F.softmax(scores.float() / sample_temperature, dim=1), 1
                ).squeeze(1)
            assert next_words.size() == (unfinished_mask.sum().item(),)

            # update generations / lengths / finished sentences / current length.
            # No need to updates the finished sequences since the value is self.pad_index by default
            generated[cur_len, unfinished_mask] = next_words

            gen_len.add_(unfinished_sents)
            generated[cur_len].masked_fill_(
                max_lengths.eq(cur_len + 1) & unfinished_sents.eq(1), self.eos_index
            )
            unfinished_sents[unfinished_mask] = (
                unfinished_sents[unfinished_mask]
                .mul(next_words.ne(self.eos_index).long())
                .mul(max_lengths[unfinished_mask].ne(cur_len + 1).long())
            )

            cur_len = cur_len + 1

            previous_unfinished_mask = unfinished_mask
            # stop when there is a </s> in each sentence, or if we exceed the maximal length
            if unfinished_sents.max() == 0:
                break

        # sanity check
        assert (generated == self.eos_index).sum() == 2 * bs

        return generated[:cur_len], gen_len

    def generate_beam(
            self,
            src_enc,
            src_len,
            tgt_lang_id,
            beam_size,
            length_penalty,
            early_stopping,
            max_len=200,
    ):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        `langs`:
            - must be None if the model only supports one language
            - lang_id if only one language is involved (LM)
            - (lang_id1, lang_id2) if two languages are involved (MT)
        """
        if isinstance(max_len, int):
            max_lengths = src_len.clone().fill_(max_len)
            global_max_len = max_len
        else:
            max_lengths = max_len
            global_max_len = int(max_lengths.max())

        # check inputs
        assert src_enc.size(0) == src_len.size(0)
        assert beam_size >= 1

        # batch size / number of words
        bs = len(src_len)
        n_words = self.n_words

        # expand to beam size the source latent representations / source lengths
        src_enc = (
            src_enc.unsqueeze(1)
            .expand((bs, beam_size) + src_enc.shape[1:])
            .contiguous()
            .view((bs * beam_size,) + src_enc.shape[1:])
        )
        src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)

        # generated sentences (batch with beam current hypotheses)
        generated = src_len.new(global_max_len, bs * beam_size)  # upcoming output
        # fill upcoming ouput with <PAD>
        generated.fill_(self.pad_index)
        # we use <EOS> for <BOS> everywhere
        generated[0].fill_(self.eos_index)

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(beam_size, global_max_len, length_penalty, early_stopping)
            for _ in range(bs)
        ]

        # positions
        positions = src_len.new(global_max_len).long()
        positions = (
            torch.arange(global_max_len, out=positions)
            .unsqueeze(1)
            .expand_as(generated)
        )

        # language IDs
        langs = positions.clone().fill_(tgt_lang_id)

        # scores for each sentence in the beam
        beam_scores = src_enc.new(bs, beam_size).float().fill_(0)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # current position
        cur_len = 1

        # cache compute states
        self.cache = {"slen": 0}

        # done sentences
        done = [False for _ in range(bs)]

        while cur_len < global_max_len:

            # compute word scores
            # TODO
            # tensor = self.forward(
            #     "fwd",
            #     x=generated[:cur_len],
            #     lengths=src_len.new(bs * beam_size).fill_(cur_len),
            #     positions=positions[:cur_len],
            #     langs=langs[:cur_len],
            #     causal=True,
            #     src_enc=src_enc,
            #     src_len=src_len,
            #     use_cache=True,
            # )
            # assert tensor.size() == (1, bs * beam_size, self.dim)
            # # (bs * beam_size, dim)
            # tensor = tensor.data[-1, :, :].type_as(src_enc)
            # scores = self.pred_layer.get_scores(tensor)  # (bs * beam_size, n_words)
            # ---
            try:
                scores = self(
                    x=generated[:cur_len],
                    lengths=src_len.new(bs * beam_size).fill_(cur_len),
                    langs=langs[:cur_len],
                    src_enc=src_enc,
                    src_len=src_len
                )
            except InvalidArgument:
                scores = self(
                    x=generated[:cur_len],
                    lengths=src_len.new(bs * beam_size).fill_(cur_len),
                    src_enc=src_enc,
                    src_len=src_len
                )
            # TODO end
            # (bs * beam_size, n_words)
            scores = F.log_softmax(scores.float(), dim=-1)
            assert scores.size() == (bs * beam_size, n_words)

            # select next words with scores
            # (bs * beam_size, n_words)
            _scores = scores + beam_scores[:, None].expand_as(scores)
            # (bs, beam_size * n_words)
            _scores = _scores.view(bs, beam_size * n_words)

            next_scores, next_words = torch.topk(
                _scores, 2 * beam_size, dim=1, largest=True, sorted=True
            )
            assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

            # next batch beam content
            # list of (bs * beam_size) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for sent_id in range(bs):

                # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(
                    next_scores[sent_id].max().item()
                )
                if done[sent_id]:
                    next_batch_beam.extend(
                        [(0, self.pad_index, 0)] * beam_size
                    )  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    # get beam and word IDs
                    beam_id = idx // n_words
                    word_id = idx % n_words

                    # end of sentence, or next word
                    if word_id == self.eos_index or cur_len + 1 == global_max_len:
                        generated_hyps[sent_id].add(
                            generated[:cur_len, sent_id * beam_size + beam_id].clone(),
                            value.item(),
                        )
                    else:
                        next_sent_beam.append(
                            (value, word_id, sent_id * beam_size + beam_id)
                        )

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break

                # update next beam content
                assert (
                    len(next_sent_beam) == 0
                    if cur_len + 1 == global_max_len
                    else beam_size
                )
                if len(next_sent_beam) == 0:
                    next_sent_beam = [
                                         (0, self.pad_index, 0)
                                     ] * beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == bs * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_len.new([x[2] for x in next_batch_beam])

            # re-order batch and internal states
            generated = generated[:, beam_idx]
            generated[cur_len] = beam_words
            for k in self.cache.keys():
                if k != "slen":
                    self.cache[k] = (
                        self.cache[k][0][beam_idx],
                        self.cache[k][1][beam_idx],
                    )

            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # visualize hypotheses
        # print([len(x) for x in generated_hyps], cur_len)
        # globals().update( locals() );
        # !import code; code.interact(local=vars())
        # for ii in range(bs):
        #     for ss, ww in sorted(generated_hyps[ii].hyp, key=lambda x: x[0], reverse=True):
        #         print("%.3f " % ss + " ".join(self.dico[x] for x in ww.tolist()))
        #     print("")

        # select the best hypotheses
        tgt_len = src_len.new(bs, beam_size)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            sorted_hyps = [
                h[1] for h in sorted(hypotheses.hyp, key=lambda x: x[0], reverse=True)
            ]
            for j, hyp in enumerate(sorted_hyps):
                tgt_len[i, j] = len(hyp) + 1
                # +1 for the <EOS> symbol
            best.append(sorted_hyps)

        # generate target batch
        decoded = src_len.new(tgt_len.max().item(), beam_size, bs).fill_(self.pad_index)
        for i, hypo_list in enumerate(best):
            for hyp_index, hypo in enumerate(hypo_list):
                decoded[: len(hypo), hyp_index, i] = hypo
                decoded[len(hypo), hyp_index, i] = self.eos_index

        # sanity check
        assert (decoded == self.eos_index).sum() == 2 * beam_size * bs

        return decoded, tgt_len, sorted([h[0] for h in hypotheses.hyp], reverse=True)


def build_onnx_model(params, dico):
    """
    Build model.
    """
    encoder = ONNXModel(dico, params.encoder_path)
    decoder = ONNXModel(dico, params.decoder_path)
    return encoder, decoder
