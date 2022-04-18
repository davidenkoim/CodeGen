from functools import lru_cache
from itertools import chain

import numpy as np

from codegen_sources.model.src.data.dictionary import OBFS


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
