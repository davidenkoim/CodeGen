from functools import lru_cache
from itertools import chain

import numpy as np

from codegen_sources.model.src.data.dictionary import OBFS
from codegen_sources.model.src.utils import batch_sentences


def get_dobf_mask(d, p, obf_type, rng, dico):
    if obf_type == "all":
        return _get_mask(d, p, rng)
    else:
        obf_type = obf_type.upper()
        type_idx = dico.obf_index[obf_type]
        obf_idxs = set(_get_obf_idxs(type_idx, OBFS[obf_type]))
        idxs_to_choose = [i for i, (v, _) in enumerate(d) if v in obf_idxs]
        if not idxs_to_choose:
            return None
        idxs_to_choose_mask = _get_mask(idxs_to_choose, p, rng)
        idxs_to_choose = [i for i, v in zip(idxs_to_choose, idxs_to_choose_mask) if not v]
        dobf_mask = np.ones(len(d), dtype=bool)
        dobf_mask[idxs_to_choose] = False
        return dobf_mask


def _get_mask(d, p, rng):
    if rng:
        mask = rng.rand(len(d)) <= p
    else:
        mask = np.random.rand(len(d)) <= p
    # make sure at least one variable is picked
    if sum(mask) == len(d):
        if rng:
            mask[rng.randint(0, len(d))] = False
        else:
            mask[np.random.randint(0, len(d))] = False
    return mask


def get_random_mapping(d, obf_type, rng, dico):
    if obf_type == "all":
        return dict(chain(*[get_random_mapping(d, t, rng, dico).items() for t in OBFS.keys()]))
    else:
        obf_type = obf_type.upper()
        type_idx = dico.obf_index[obf_type]
        obf_idxs = _get_obf_idxs(type_idx, OBFS[obf_type])
        obf_idxs_set = set(obf_idxs)
        typed_idxs = [k for k, _ in d if k in obf_idxs_set]
        rnd_idxs = rng.choice(obf_idxs, size=len(typed_idxs), replace=False) if rng \
            else np.random.choice(obf_idxs, size=len(typed_idxs), replace=False)
        return dict(zip(typed_idxs, rnd_idxs))


@lru_cache()
def _get_obf_idxs(idx, number):
    return [str(i) for i in range(idx, idx + number)]


def deobfuscate_by_variable(x, y, p, roberta_mode, dico, pad_index, eos_index, max_len,
                            rng=None, obf_type="all", shuffle_masks=True):
    obf_tokens = (x >= dico.obf_index["CLASS"]) * (
            x < (dico.obf_index["CLASS"] + dico.n_obf_tokens)
    )
    x[obf_tokens] = -x[obf_tokens]

    # convert sentences to strings and dictionary to a python dictionary (obf_token_special , original_name)
    x_ = [
        " ".join(
            [
                str(w)
                for w in s
                if w not in [pad_index, eos_index]
            ]
        )
        for s in x.transpose(0, 1).tolist()
    ]
    y_ = [
        " ".join(
            [
                str(w)
                for w in s
                if w not in [pad_index, eos_index]
            ]
        )
        for s in y.transpose(0, 1).tolist()
    ]

    # filter out sentences without identifiers
    xy = tuple(zip(*[(xi, yi) for xi, yi in zip(x_, y_) if yi]))
    x_, y_ = (list(xy[0]), list(xy[1])) if len(xy) == 2 else ([], [])

    if roberta_mode:
        sep = (
            f" {dico.word2id['Ġ|']} {dico.word2id['Ġ']} "
        )
    else:
        sep = f" {dico.word2id['|']} "
    # reversed order to have longer obfuscation first, to make replacement in correct order
    d = [
        list(
            reversed(
                [
                    (
                        mapping.strip().split()[0],
                        " ".join(mapping.strip().split()[1:]),
                    )
                    for mapping in pred.split(sep)
                ]
            )
        )
        for pred in y_
    ]

    # restore x i.e. select variable with probability p and restore all occurence of this variable
    # keep only unrestored variable in dictionary d_
    x = []
    y = []

    for i, di in enumerate(d):
        d_ = []
        dobf_mask = get_dobf_mask(di, p, obf_type, rng, dico)
        if dobf_mask is None:
            continue
        # shuffle masks
        random_mapping = get_random_mapping(di, obf_type, rng, dico) if shuffle_masks else {k: k for k, _ in di}
        for m, (k, v) in enumerate(di):
            if dobf_mask[m]:
                x_[i] = x_[i].replace(f"-{k}", f"{v}")
            else:
                d_.append((random_mapping[k], v))
                x_[i] = x_[i].replace(f"-{k}", f"{random_mapping[k]}")
        if roberta_mode:
            # we need to remove the double space introduced during deobfuscation, i.e the "Ġ Ġ"
            sent_ids = np.array(
                [
                    dico.word2id[index]
                    for index in (
                    " ".join(
                        [
                            dico.id2word[int(w)]
                            for w in x_[i].split()
                        ]
                    ).replace("Ġ Ġ", "Ġ")
                ).split()
                ]
            )
        else:
            sent_ids = np.array([int(id) for id in x_[i].split()])
        if len(sent_ids) < max_len:
            x.append(sent_ids)
            d_ids = sep.join([" ".join([k, v]) for k, v in reversed(d_)])
            d_ids = np.array([int(id) for id in d_ids.split()])
            y.append(d_ids)

    if len(x) == 0:
        return None, None, None, None

    x, len_x = batch_sentences(x, pad_index, eos_index)
    y, len_y = batch_sentences(y, pad_index, eos_index)

    assert sum(sum((x < 0).float())) == 0

    return x, len_x, y, len_y
