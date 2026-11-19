"""Metrics for the change-detection QA datasets ``vlmcd_<task>_<split>``.

``VQAEvaluator`` calls ``vlmcd_metrics`` after each evaluation; ``vlm_cd/score_vlm_cd.py`` in
the parent repo calls it on saved ``*_predictions.json`` files without a GPU. The module
needs only numpy, scipy and ``datasets/vlm_cd.py``, so it can also be loaded by file path
(the ``qwen3d`` package itself takes minutes to import on a network file system).

Metric groups (``describe``):

- **changes**: precision / recall / F1 of ``(action, class)`` pairs, and per-action F1.
- **distances** (``move_dist_*``): for moved objects whose true distance is given (seen in
  both scans), paired with predicted moves of the same class.
- **volumes** (``volume_*``): for every true entry with a ``volume_m3`` (``VLMCD.ANSWER_VOLUME``),
  paired with predicted entries of the same action and class. Volumes span five orders of
  magnitude, so errors are in log10 units or as ratios.
- **order** (``order_*``, ``VLMCD.ANSWER_ORDER size`` only): whether the prediction lists
  moved, then added, then removed objects, each largest first.
- **format** (``json_*``): whether predictions are complete JSON.

``count``: per-action accuracy and absolute / signed errors of the counts. See
``vlmcd_metrics`` for the definition of each key.
"""

import math
from collections import Counter, defaultdict

import numpy as np

try:
    from .datasets import vlm_cd
except ImportError:  # loaded by file path, with datasets/ on sys.path (score_vlm_cd.py)
    import vlm_cd

DIST_THRESHOLDS = (0.1, 0.25, 0.5)  # metres
VOLUME_RATIOS = (1.25, 2.0, 4.0)    # max(pred / true, true / pred)
VOLUME_FLOOR = 1e-7                 # m^3; keeps logs finite for a 0 prediction


def match_values(pred, gt, key, actions=None, log=False):
    """Pair predicted and true values of ``key`` within each ``(action, class)``.

    Inside a group, predicted entries that give the value and true entries that have it
    are paired one to one, minimising the total absolute error
    (``scipy.optimize.linear_sum_assignment``; with one object per class, the usual case,
    this is simply the pair).

    Parameters
    ----------
    pred, gt : list of dict
        ``vlm_cd.parse_describe_entries`` output.
    key : str
        ``distance_m`` or ``volume_m3``.
    actions : tuple of str, optional
        Only these actions (default: all).
    log : bool
        Pair by error in log10 units (volumes).

    Returns
    -------
    tuple
        ``(pairs, n_gt, n_missing)``: ``(pred, true)`` value pairs; true entries with the
        value; and of those, ones that a predicted entry of the group could have matched
        but which gave no value.
    """
    from scipy.optimize import linear_sum_assignment

    def f(v):
        return math.log10(max(v, VOLUME_FLOOR)) if log else v

    pairs, n_gt, n_missing = [], 0, 0
    groups = {(e["action"], e["category"]) for e in gt
              if e[key] is not None and (actions is None or e["action"] in actions)}
    for grp in groups:
        g = [e[key] for e in gt if (e["action"], e["category"]) == grp and e[key] is not None]
        p_all = [e[key] for e in pred if (e["action"], e["category"]) == grp]
        p = [v for v in p_all if v is not None]
        n_gt += len(g)
        if p:
            cost = np.abs(np.array([f(v) for v in p])[:, None] - np.array([f(v) for v in g])[None, :])
            rows, cols = linear_sum_assignment(cost)
            pairs += [(p[r], g[c]) for r, c in zip(rows, cols)]
        n_missing += max(0, min(len(g), len(p_all)) - len(p))
    return pairs, n_gt, n_missing


def gt_positions(pred, gt):
    """For each predicted entry, the index of the true entry it stands for (or None).

    The k-th prediction of an ``(action, class)`` stands for the k-th true entry of that
    group (true entries are already in the target order).
    """
    index = defaultdict(list)
    for j, e in enumerate(gt):
        index[(e["action"], e["category"])].append(j)
    seen, out = Counter(), []
    for e in pred:
        k = (e["action"], e["category"])
        out.append(index[k][seen[k]] if seen[k] < len(index[k]) else None)
        seen[k] += 1
    return out


def order_stats(pred, gt):
    """Size-order checks of one ``describe`` prediction.

    Returns
    -------
    dict
        ``grouped`` (bool: actions appear moved, added, removed), ``concordant`` /
        ``pairs`` (same-action pairs of matched entries, listed in the true relative
        order / all; pairs of equal true volume are skipped), and ``self_ok`` (bool, or
        None when not checkable: within each action, the predicted volumes never increase).
    """
    rank = [vlm_cd.ACTIONS.index(e["action"]) for e in pred]
    grouped = all(a <= b for a, b in zip(rank, rank[1:]))
    pos = gt_positions(pred, gt)
    concordant = pairs = 0
    for i in range(len(pred)):
        for j in range(i + 1, len(pred)):
            if pos[i] is None or pos[j] is None or pred[i]["action"] != pred[j]["action"]:
                continue
            vi, vj = gt[pos[i]]["volume_m3"], gt[pos[j]]["volume_m3"]
            if vi is not None and vi == vj:
                continue  # a tie: either order is right
            pairs += 1
            concordant += pos[i] < pos[j]
    self_ok, checked = True, False
    for a, b in zip(pred, pred[1:]):
        if a["action"] == b["action"] and a["volume_m3"] is not None and b["volume_m3"] is not None:
            checked = True
            self_ok &= a["volume_m3"] >= b["volume_m3"]
    return {"grouped": grouped, "concordant": concordant, "pairs": pairs,
            "self_ok": self_ok if checked else None}


def _truncated(text):
    """A JSON array the generation limit cut off (starts like one, does not parse)."""
    t = text.strip()
    return t.startswith("[") and not vlm_cd.is_json_answer(t)


def distance_metrics(pairs, n_gt, n_missing):
    """``move_dist_*`` keys from ``match_values(..., "distance_m", ("move",))``."""
    if not n_gt:
        return {}
    err = np.array([p - g for p, g in pairs]) if pairs else np.array([np.nan])
    ae = np.abs(err)
    out = {
        "move_dist_mae_m": float(np.mean(ae)),
        "move_dist_medae_m": float(np.median(ae)),
        "move_dist_rmse_m": float(np.sqrt(np.mean(err ** 2))),
        "move_dist_bias_m": float(np.mean(err)),
        "move_dist_abs_rel": float(np.mean([abs(p - g) / g for p, g in pairs if g > 0]))
        if any(g > 0 for _, g in pairs) else float("nan"),
    }
    for t in DIST_THRESHOLDS:
        out[f"move_dist_acc_{t}m"] = float(np.mean(ae <= t)) if pairs else 0.0
    out["move_dist_recall_0.25m"] = float(np.sum(ae <= 0.25)) / n_gt if pairs else 0.0
    out["move_dist_pairs"] = float(len(pairs))
    out["move_dist_missing"] = float(n_missing)
    return out


def volume_metrics(pairs, n_gt, n_missing):
    """``volume_*`` keys from ``match_values(..., "volume_m3", log=True)``."""
    if not n_gt:
        return {}
    if not pairs:
        return {"volume_pairs": 0.0, "volume_missing": float(n_missing),
                "volume_recall_x2": 0.0}
    p = np.maximum(np.array([x for x, _ in pairs]), VOLUME_FLOOR)
    g = np.maximum(np.array([y for _, y in pairs]), VOLUME_FLOOR)
    le = np.log10(p) - np.log10(g)
    ratio = np.maximum(p / g, g / p)
    out = {
        "volume_mae_log10": float(np.mean(np.abs(le))),
        "volume_medae_log10": float(np.median(np.abs(le))),
        "volume_rmse_log10": float(np.sqrt(np.mean(le ** 2))),
        "volume_bias_log10": float(np.mean(le)),
        "volume_abs_rel": float(np.mean(np.abs(p - g) / g)),
        "volume_med_abs_rel": float(np.median(np.abs(p - g) / g)),
    }
    for r in VOLUME_RATIOS:
        out[f"volume_acc_x{r:g}"] = float(np.mean(ratio <= r))
    out["volume_recall_x2"] = float(np.sum(ratio <= 2.0)) / n_gt
    if len(pairs) >= 3:
        from scipy.stats import spearmanr
        rho = spearmanr(p, g).correlation
        out["volume_spearman"] = float(rho) if rho == rho else float("nan")
    out["volume_pairs"] = float(len(pairs))
    out["volume_missing"] = float(n_missing)
    return out


def describe_metrics(res, order="size"):
    """Metrics of ``describe`` predictions; see ``vlmcd_metrics``."""
    out = {}
    tp = fp = fn = 0
    per = {a: [0, 0, 0] for a in vlm_cd.ACTIONS}
    dist, vol = ([], 0, 0), ([], 0, 0)
    n_json = n_trunc = 0
    json_task = False
    n_grouped = n_ordered_preds = n_conc = n_pairs = n_exact = n_exact_preds = n_self = n_self_preds = 0
    for pred, answers in res:
        pr, gr = vlm_cd.parse_describe_entries(pred), vlm_cd.parse_describe_entries(answers[0])
        p = Counter((e["action"], e["category"]) for e in pr)
        g = Counter((e["action"], e["category"]) for e in gr)
        for key in set(p) | set(g):
            t = min(p[key], g[key])
            tp, fp, fn = tp + t, fp + p[key] - t, fn + g[key] - t
            per[key[0]][0] += t
            per[key[0]][1] += p[key] - t
            per[key[0]][2] += g[key] - t
        d = match_values(pr, gr, "distance_m", actions=("move",))
        v = match_values(pr, gr, "volume_m3", log=True)
        dist = (dist[0] + d[0], dist[1] + d[1], dist[2] + d[2])
        vol = (vol[0] + v[0], vol[1] + v[1], vol[2] + v[2])
        if vlm_cd.is_json_answer(answers[0]):
            json_task = True
            n_json += vlm_cd.is_json_answer(pred)
            n_trunc += _truncated(pred)
        if order == "size" and pr:
            s = order_stats(pr, gr)
            n_ordered_preds += 1
            n_grouped += s["grouped"]
            n_conc += s["concordant"]
            n_pairs += s["pairs"]
            if s["pairs"]:
                n_exact_preds += 1
                n_exact += s["concordant"] == s["pairs"]
            if s["self_ok"] is not None:
                n_self_preds += 1
                n_self += s["self_ok"]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out.update(change_precision=prec, change_recall=rec,
               change_f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    for a, (t, f_p, f_n) in per.items():
        out[f"{a}_f1"] = 2 * t / (2 * t + f_p + f_n) if t + f_p + f_n else 0.0
    out.update(distance_metrics(*dist))
    out.update(volume_metrics(*vol))
    if n_ordered_preds:
        out["order_grouped"] = n_grouped / n_ordered_preds
        if n_pairs:
            out["order_pair_acc"] = n_conc / n_pairs
            out["order_exact"] = n_exact / n_exact_preds
            out["order_pairs"] = float(n_pairs)
        if n_self_preds:
            out["order_follows_pred_volume"] = n_self / n_self_preds
    if json_task:
        out["json_valid"] = n_json / max(len(res), 1)
        out["json_truncated"] = n_trunc / max(len(res), 1)
    return out


def count_metrics(res):
    """Metrics of ``count`` predictions; see ``vlmcd_metrics``."""
    acts = vlm_cd.ACTIONS
    correct = {a: 0 for a in acts}
    abs_err = {a: 0.0 for a in acts}
    all_correct = parsed = n_json = 0
    json_task = False
    tot_err = []
    for pred, answers in res:
        p, g = vlm_cd.parse_count(pred), vlm_cd.parse_count(answers[0])
        ok = [p[a] == g[a] for a in acts]
        for a, o in zip(acts, ok):
            correct[a] += o
            abs_err[a] += abs((p[a] or 0) - (g[a] or 0))
        all_correct += all(ok)
        parsed += all(p[a] is not None for a in acts)
        tot_err.append(sum(p[a] or 0 for a in acts) - sum(g[a] or 0 for a in acts))
        if vlm_cd.is_json_answer(answers[0]):
            json_task = True
            n_json += vlm_cd.is_json_answer(pred)
    n = max(len(res), 1)
    out = {f"{a}_count_acc": correct[a] / n for a in acts}
    out.update({f"{a}_count_mae": abs_err[a] / n for a in acts})
    tot = np.array(tot_err, dtype=float) if tot_err else np.zeros(1)
    out.update(count_all_acc=all_correct / n, total_count_mae=float(np.mean(np.abs(tot))),
               total_count_bias=float(np.mean(tot)), count_parsed=parsed / n)
    if json_task:
        out["json_valid"] = n_json / n
    return out


def vlmcd_metrics(dataset_name, res, order="size"):
    """Task metrics for ``vlmcd_<task>_<split>`` predictions.

    Parameters
    ----------
    dataset_name : str
    res : list of tuple
        ``(prediction, [answer])`` pairs, lower-cased.
    order : str
        ``VLMCD.ANSWER_ORDER`` the targets were built with; the ``order_*`` metrics are
        only computed for ``size``.

    Returns
    -------
    dict
        ``describe``:

        - ``change_precision`` / ``change_recall`` / ``change_f1``: micro scores over
          ``(action, class)`` pairs; ``move_f1`` / ``add_f1`` / ``remove_f1`` per action.
        - ``move_dist_*``, over moved objects whose true distance is given, each paired with
          a predicted move of the class (``match_values``): ``mae_m``, ``medae_m``,
          ``rmse_m``, ``bias_m`` (mean predicted minus true: > 0 overestimates),
          ``abs_rel`` (mean |error| / true), ``acc_<t>m`` (share of pairs within t m),
          ``recall_0.25m`` (share of all such true moves found with an error <= 0.25 m),
          ``pairs`` and ``missing`` (paired moves predicted without a distance).
        - ``volume_*``, over true entries with a volume, paired with predicted entries of
          the same action and class: ``mae_log10`` / ``medae_log10`` / ``rmse_log10``
          (error in orders of magnitude), ``bias_log10`` (> 0 overestimates),
          ``abs_rel`` / ``med_abs_rel`` (|error| / true), ``acc_x<r>`` (share of pairs
          within a factor r), ``recall_x2`` (share of all true volumes found within a
          factor 2), ``spearman`` (rank correlation of predicted and true volumes over
          all pairs, >= 3 pairs), ``pairs``, ``missing``.
        - ``order_*`` (``order`` ``size``): ``grouped`` (share of predictions listing
          moved, then added, then removed), ``pair_acc`` (share of same-action pairs of
          matched entries in the true order, i.e. largest first; true ties skipped),
          ``exact`` (share of predictions with such pairs that order all of them right),
          ``pairs``, and ``follows_pred_volume`` (share of predictions whose own
          ``volume_m3`` values never increase within an action).
        - ``json_valid`` (share of predictions that parse as a whole) and
          ``json_truncated`` (share that start a JSON array but do not parse, usually the
          generation limit), for JSON targets.

        ``count``: ``<action>_count_acc`` and ``<action>_count_mae``, ``count_all_acc``
        (all three right), ``total_count_mae`` and ``total_count_bias`` (predicted minus
        true total), ``count_parsed`` (share with all three counts readable), and
        ``json_valid`` for JSON targets.
    """
    task = vlm_cd.dataset_task(dataset_name)
    if task == "describe":
        out = describe_metrics(res, order)
    elif task == "count":
        out = count_metrics(res)
    else:
        out = {}
    print(", ".join(f"{k}: {v:.4f}" for k, v in out.items()))
    return out
