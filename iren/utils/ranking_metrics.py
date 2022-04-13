import os
from typing import List, Tuple, Dict

from codegen_sources.model.src.utils import read_file_lines


def run_ranking_scoring(ref_path, hyp_paths):
    for h in hyp_paths:
        assert os.path.isfile(h), f"file {h} does not exist"
    assert os.path.isfile(ref_path) or os.path.isfile(ref_path + "0")
    refs: List[str] = read_file_lines(ref_path)
    hyps_list: List[Tuple[str]] = list(zip(*[read_file_lines(path) for path in hyp_paths]))
    return compute_ranking_metrics(refs, hyps_list)


def compute_ranking_metrics(refs: List[str], hyps_list: List[Tuple[str]]) -> Dict[str, float]:
    total = len(refs)
    assert total > 0
    mrr, top1, top5 = 0, 0, 0
    for ref, hyps in zip(refs, hyps_list):
        if ref in hyps:
            idx = hyps.index(ref)
            top1 += idx < 1
            top5 += idx < 5
            mrr += 1 / (1 + idx)
    return {
        "mrr": mrr / total,
        "top1": top1 / total,
        "top5": top5 / total
    }
