# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
import logging
import torch
import random
import detectron2.utils.comm as comm
import numpy as np
from tabulate import tabulate
from prettytable import PrettyTable
from detectron2.evaluation import DatasetEvaluator
from qwen3d.utils.misc import all_gather, is_dist_avail_and_initialized
from torch.nn import functional as F
from qwen3d.data_video.referrential_grounding_evaluator import ReferrentialGroundingEvaluator
# from pycocoevalcap.bleu.bleu import Bleu
# from pycocoevalcap.meteor.meteor import Meteor
# from pycocoevalcap.rouge.rouge import Rouge
# from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import re
from collections import Counter

import ipdb
st = ipdb.set_trace


from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

tokenizer = PTBTokenizer()

# Change-detection metrics live in their own module so vlm_cd/score_vlm_cd.py can load them
# without importing the package.
from qwen3d.data_video.vlmcd_metrics import vlmcd_metrics  # noqa: E402


class VQAEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        evaluate_detection,
        cfg=None
    ):
        self._logger = logging.getLogger(__name__)
        self.dataset_name = dataset_name
        self.cfg = cfg
        self.num_viz = 0
        self.results = []
        self.vlmcd_records = []
        self.metric2scorer = {'cider': Cider(), 'bleu': Bleu(), 'meteor': Meteor(), 'rouge': Rouge()}

        self.evaluate_detection = evaluate_detection
        print(f"evaluates grounding: {self.evaluate_detection}")
        if self.evaluate_detection:
            self.detection_evaluator = ReferrentialGroundingEvaluator(
                dataset_name,
                thresholds=[0.25, 0.5, 0.75],
                topks=[1, 2, 5],
                cfg=cfg
            )

        self._cpu_device = torch.device("cpu")


    def reset(self):
        self.results = []
        self.vlmcd_records = []
        self.num_viz = 0
        if self.evaluate_detection:
            self.detection_evaluator.reset()

    def _save_vlmcd_predictions(self, eval_results):
        """Write per-example predictions and the metrics to the inference folder.

        ``inference/<dataset>_predictions.json`` always holds the latest eval. During
        training (a detectron2 event storage is active) the same file is also kept as
        ``inference/iter_<iteration:07d>/<dataset>_predictions.json``, so the periodic
        evals of earlier checkpoints are not overwritten; the qualitative viewer
        (visualization/training_runs/qualitative_results) lists each as a checkpoint.
        """
        import json
        import os
        root = getattr(self.cfg, "OUTPUT_DIR", None) if self.cfg is not None else None
        if not root:
            return
        try:
            from detectron2.utils.events import get_event_storage
            iteration = int(get_event_storage().iter)
        except (AssertionError, ImportError, AttributeError):  # eval-only: no storage
            iteration = None
        payload = {"metrics": eval_results, "settings": self._vlmcd_settings(),
                   "iteration": iteration, "predictions": self.vlmcd_records}
        out_dir = os.path.join(root, "inference")
        dirs = [out_dir] + ([os.path.join(out_dir, f"iter_{iteration:07d}")] if iteration is not None else [])
        for d in dirs:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{self.dataset_name}_predictions.json"), "w") as f:
                json.dump(payload, f, indent=1)
        if iteration is not None:
            self._prune_vlmcd_evals(out_dir)

    def _prune_vlmcd_evals(self, out_dir):
        """Keep the newest ``VLMCD.KEEP_EVALS`` ``inference/iter_*`` folders (0: all)."""
        import os
        import re
        import shutil
        vc = getattr(self.cfg, "VLMCD", None)
        keep = int(vc.get("KEEP_EVALS", 0) or 0) if vc else 0
        if keep <= 0:
            return
        found = sorted((int(m.group(1)), name) for name in os.listdir(out_dir)
                       if (m := re.match(r"^iter_(\d+)$", name)) and os.path.isdir(os.path.join(out_dir, name)))
        for _, name in found[:-keep]:
            path = os.path.join(out_dir, name)
            shutil.rmtree(path, ignore_errors=True)
            self._logger.info(f"Deleted old eval {path} (VLMCD.KEEP_EVALS {keep})")

    def _vlmcd_settings(self):
        """The answer settings the targets were built with (read back by score_vlm_cd.py)."""
        vc = getattr(self.cfg, "VLMCD", None) if self.cfg is not None else None
        if not vc:
            return {}
        out = {k: vc.get(k) for k in ("ANSWER_FORMAT", "ANSWER_ORDER", "ANSWER_VOLUME", "ORDER_IN_PROMPT",
                                       "ANSWER_POSE", "ANSWER_BBOX",
                                       "STATE_ENCODING", "ONLY_VISIBLE_CHANGES", "FRAMES_PER_STATE")}
        gen = self.cfg.get("GEN_KWARGS")
        out["MAX_NEW_TOKENS"] = gen.get("max_new_tokens") if gen else None  # json_compliance_vlm_cd.py
        return out

    # called per batch
    def process(self, inputs, outputs):
        assert(len(inputs) == 1)
        for i in range(len(outputs[0])):
            if 'sr3d_data' in inputs[0]:
                if self.dataset_name == "mmlupro_text_bench":
                    self.results.append((outputs[0][i].lower(), inputs[i]['answer'].lower()))
                elif self.dataset_name == 'realworld_vqa_bench':
                    self.results.append((outputs[0][i].lower(), [answer.lower() for answer in inputs[0]['sr3d_data'][i]['answers']]))
                elif 'scanqa' in self.dataset_name or 'sqa3d' in self.dataset_name:
                    self.results.append((outputs[0][i].lower(), [answer.lower() for answer in inputs[0]['sr3d_data'][i]['answers']]))
                elif self.dataset_name.startswith('vlmcd_'):
                    self.results.append((outputs[0][i].lower().strip(), [answer.lower() for answer in inputs[0]['sr3d_data'][i]['answers']]))
                    self.vlmcd_records.append({
                        "example_id": inputs[0]['sr3d_data'][i]['example_id'],
                        "frame_ids": inputs[0].get('frame_ids'),
                        "prediction": outputs[0][i].strip(),
                        "answer": inputs[0]['sr3d_data'][i]['answers'][0],
                    })
            else:
                if self.dataset_name == "mmlupro_text_bench":
                    self.results.append((outputs[0][i].lower(), inputs[i]['answer'].lower()))
                elif self.dataset_name == 'realworld_vqa_bench':
                    self.results.append((outputs[0][i].lower(), inputs[i]['answer'].lower()))
                elif 'scanqa' in self.dataset_name or 'sqa3d' in self.dataset_name:
                    self.results.append((outputs[0][i].lower(), inputs[i]['answer'].lower()))
        if self.evaluate_detection:
            self.detection_evaluator.process(inputs, outputs[1])

    # called per dataset
    def evaluate(self):
        if is_dist_avail_and_initialized():
            # InferenceSampler gives each rank a contiguous slice; concatenate
            # every rank's list (not just rank 0's) to score the whole split.
            res = list(itertools.chain(*all_gather(self.results)))
            self.vlmcd_records = list(itertools.chain(*all_gather(self.vlmcd_records)))
            comm.synchronize()
            if self.evaluate_detection:
                detection_result = self.detection_evaluator.evaluate()
            if not comm.is_main_process():
                comm.synchronize()
                return {}
        else:
            res = self.results
            if self.evaluate_detection:
                detection_result = self.detection_evaluator.evaluate()

        gts = {}
        preds = {}
        eval_results = {}
        exact_matches = 0
        tokenizer = PTBTokenizer()

        for i in range(len(res)):
            if "bench" in self.dataset_name:
                res[i] = (res[i][0].lower().strip().rstrip("."), res[i][1].lower().strip().rstrip("."))
                if res[i][0] in res[i][1]:
                    exact_matches += 1
                else:
                    print(f"Incorrect: Predicted: {res[i][0]}, Ground Truth: {res[i][1]}")
            elif self.dataset_name.startswith('vlmcd_'):
                # Long answers: count exact matches without printing every miss.
                exact_matches += res[i][0].rstrip(".") in [a.rstrip(".") for a in res[i][1]]
            elif 'scanqa' in self.dataset_name or 'sqa3d' in self.dataset_name:
                if res[i][0] in res[i][1]:
                    exact_matches += 1
                else:
                    print(f"Incorrect: Predicted: {res[i][0]}, Ground Truth: {res[i][1]}")
                
            preds[i] = [{'caption': res[i][0]}]
            gts[i] = [{'caption': ans} for ans in res[i][1]]
        

        gts = tokenizer.tokenize(gts)
        preds = tokenizer.tokenize(preds)

        eval_results['exact_match'] = exact_matches / len(res)
        print(f'exact match: {np.round(exact_matches / len(res), 4)}, ')

        for metric_name in self.metric2scorer.keys():
            # Report the scorer's corpus-level score (standard pycocoevalcap protocol).
            # The mean of per-sentence scores is wrong for BLEU-4 (~0 for the 1-3 word
            # answers here) and is not METEOR's aggregate. CIDEr/ROUGE are unaffected.
            score = self.metric2scorer[metric_name].compute_score(gts, preds)[0]
            if metric_name == 'bleu':
                score = score[-1] # bleu-4
            print(f'{metric_name}: {np.round(score, 4)}, ', end='')
            eval_results[metric_name] = float(score)
        print()

        if self.evaluate_detection:
            eval_results.update(detection_result)

        if self.dataset_name.startswith('vlmcd_'):
            order = self._vlmcd_settings().get("ANSWER_ORDER") or "size"
            eval_results.update(vlmcd_metrics(self.dataset_name, res, order))
            self._save_vlmcd_predictions(eval_results)

        if is_dist_avail_and_initialized():
            comm.synchronize()

        return eval_results
