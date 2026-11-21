"""Change-detection QA on ``DR8803VL/vlm-change-detection-4k`` (MolmoSpaces point cloud difference).

Each example is one iTHOR room rendered twice along the same camera path: before (S0) and
after (S1) several objects were moved, added or removed. The frames (RGB, depth, camera
pose) are streamed from the Hugging Face Hub by ``VlmCdDatasetMapper``; this module only
holds the light-weight parts:

- the **index** (``$VLMCD_DATA_DIR/index.json``), built once by
  ``vlm_cd/prepare_vlm_cd.py`` in the parent repo. It stores, per example, the changes,
  the camera intrinsics and where the example's frames sit in the ``walkthrough_frames``
  shards, so the mapper reads exactly two parquet row groups per sample;
- the **splits** (``$VLMCD_DATA_DIR/{train,val,test}.txt``), one example id per line;
- the **questions and answers** built from the changes that are visible in the frames a
  sample uses;
- the detectron2 **registration** of ``vlmcd_<task>_<split>`` datasets.

Tasks:

``describe``
    "List every object that was moved, added or removed." With ``VLMCD.ANSWER_FORMAT json``
    (the default) the answer is a JSON array with one entry per change, e.g.
    ``[{"action":"moved","class":"vase","volume_m3":0.00213,"distance_m":6.4},
    {"action":"removed","class":"statue","volume_m3":0.0151}]``
    (no support surface, unlike ``text``: it is not scored, and leaving it out keeps
    answers under the generation limit). ``volume_m3`` (``VLMCD.ANSWER_VOLUME``, the
    object-frame bounding-box volume from ``prepare_vlm_cd.py``, 3 significant digits) is
    the key ``VLMCD.ANSWER_ORDER size`` sorts by, so the target shows why entries come in
    their order. ``distance_m`` (how far
    the object's origin travelled, from the dataset's ``displacement_meters``) is given
    only for moved objects seen in both scans, the only ones whose distance the model can
    measure. With ``text`` it is one clause per change, e.g. ``moved the vase on the
    diningtable by 6.4 m; removed the statue from the shelfmesh``.
``count``
    "How many objects were moved, added and removed?" e.g. ``{"moved":7,"added":1,"removed":1}``
    (``json``) or ``7 moved, 1 added, 1 removed`` (``text``).
"""

import json
import math
import os
import re
from collections import Counter
from decimal import Decimal

from detectron2.data import DatasetCatalog, MetadataCatalog

REPO = "DR8803VL/vlm-change-detection-4k"
TASKS = ("describe", "count")
SPLITS = ("train", "val", "test")
ACTIONS = ("move", "add", "remove")
PAST = {"move": "moved", "add": "added", "remove": "removed"}
PRESENT = {v: k for k, v in PAST.items()}
ANSWER_FORMATS = ("json", "text")
DISTANCE_DECIMALS = 2  # centimetres: a tenth of moves travel under 0.2 m
POSES = ("none", "orient", "finalpose")  # VLMCD.ANSWER_POSE
POSE_DECIMALS = 2      # centimetres for the translation of final_pose; angles use one decimal
VOLUME_DIGITS = 3      # significant digits of volume_m3 (objects span ~1e-5 to ~1 m^3)

_INDEX_CACHE = {}


def data_dir():
    """Directory holding ``index.json`` and the split files.

    Returns
    -------
    str
        ``$VLMCD_DATA_DIR``, or ``data/vlm_cd`` next to the qwen3d checkout.
    """
    default = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "vlm_cd")
    return os.path.abspath(os.environ.get("VLMCD_DATA_DIR", default))


def load_index(root=None):
    """Load (and memoise) the example index.

    Parameters
    ----------
    root : str, optional
        Directory with ``index.json``. Defaults to ``data_dir()``.

    Returns
    -------
    dict
        ``{"repo", "revision", "examples": {example_id: record}}``.
    """
    root = root or data_dir()
    if root not in _INDEX_CACHE:
        path = os.path.join(root, "index.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found; build it with `python vlm_cd/prepare_vlm_cd.py` "
                f"or set VLMCD_DATA_DIR")
        with open(path) as f:
            index = json.load(f)
        # The index keeps the raw dataset fields; names are cleaned here, in one place.
        for rec in index["examples"].values():
            for c in rec["changes"]:
                c["category"] = clean_category(c.get("raw_category"), c["object"])
                c["surface"] = clean_surface(c.get("support_surface"))
        _INDEX_CACHE[root] = index
    return _INDEX_CACHE[root]


def load_split(split, root=None):
    """Example ids of one split.

    Parameters
    ----------
    split : str
        ``"train"``, ``"val"`` or ``"test"``.
    root : str, optional
        Directory with the split files. Defaults to ``data_dir()``.

    Returns
    -------
    list of str
    """
    root = root or data_dir()
    with open(os.path.join(root, f"{split}.txt")) as f:
        return [line.strip() for line in f if line.strip()]


def clean_surface(name):
    """Readable support-surface name.

    Parameters
    ----------
    name : str or None
        E.g. ``"diningtable_a1271102e7d3313087397a337003b3b7_1_0_0"`` or
        ``"standardwindow1(3)(2)_..."`` or ``"fp301:standardwindow_..."``.

    Returns
    -------
    str or None
        E.g. ``"diningtable"``, ``"standardwindow"``; ``None`` when there is no surface.
    """
    if not name:
        return None
    base = name.split("_")[0].split(":")[-1]  # "fp301:standardwindow..." -> "standardwindow"
    base = re.sub(r"\(.*?\)", "", base)
    base = re.sub(r"\d+$", "", base).strip()
    return base or None


def clean_category(category, obj=None):
    """Readable object category.

    Parameters
    ----------
    category : str or None
        The change's ``category``.
    obj : str, optional
        The change's ``object`` (``"vase_1"``), used when ``category`` is missing.

    Returns
    -------
    str
        Lower-case letters and digits only, so ``parse_describe`` can read it back.
    """
    name = category or re.sub(r"_\d+$", "", obj or "object")
    return re.sub(r"[^a-z0-9]", "", name.lower()) or "object"


def change_clause(change):
    """One clause of a ``describe`` answer.

    Parameters
    ----------
    change : dict
        An index change record: ``action``, ``category``, ``surface``, ``displacement_m``.

    Returns
    -------
    str
    """
    cat, surf = change["category"], change["surface"]
    if change["action"] == "move":
        text = f"moved the {cat}"
        if surf:
            text += f" on the {surf}"
        if change.get("displacement_m") is not None:
            text += f" by {change['displacement_m']:.1f} m"
        return text
    if change["action"] == "add":
        return f"added a {cat}" + (f" on the {surf}" if surf else "")
    return f"removed the {cat}" + (f" from the {surf}" if surf else "")


def has_distance(change):
    """Whether a ``json`` answer gives this change's distance.

    Parameters
    ----------
    change : dict
        An index change record; ``seen_before`` / ``seen_after`` are added by the mapper's
        ``visible_changes`` (missing means seen).

    Returns
    -------
    bool
        True for a moved object with a known displacement that was seen at both its old
        and its new place.
    """
    return (change["action"] == "move" and change.get("displacement_m") is not None
            and change.get("seen_before", True) and change.get("seen_after", True))


def round_significant(x, digits=VOLUME_DIGITS):
    """``x`` rounded to ``digits`` significant digits (0 stays 0)."""
    x = float(x)
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, digits - 1 - math.floor(math.log10(abs(x))))


def change_record(change, volume=True, pose="none", bbox=False):
    """One entry of a ``json`` ``describe`` answer.

    Parameters
    ----------
    change : dict
        An index change record.
    volume : bool
        Add ``volume_m3`` (``VLMCD.ANSWER_VOLUME``).
    pose : str
        ``VLMCD.ANSWER_POSE``. ``orient`` adds ``rotation_deg`` (how far the object turned,
        unsigned); ``finalpose`` adds that and ``final_pose``, the object's new pose relative
        to its old one: ``translation_m`` (world-frame ``[dx, dy, dz]``) and ``yaw_deg``
        (signed turn about the vertical axis). Both only where ``has_distance``.
    bbox : bool
        ``VLMCD.ANSWER_BBOX``: add ``bbox_before`` (moved and removed objects seen before)
        and ``bbox_after`` (moved and added objects seen after), each the world axis-aligned
        box ``{"center": [x, y, z], "size": [dx, dy, dz]}`` in metres.

    Returns
    -------
    dict
        ``action`` (``moved`` / ``added`` / ``removed``), ``class``, ``volume_m3`` (when
        ``volume`` and the index knows the object's size), and ``distance_m`` when
        ``has_distance``, then ``rotation_deg`` and ``final_pose`` per ``pose``, then the
        boxes per ``bbox``.

    Raises
    ------
    KeyError
        ``volume`` is set but the index was built without object sizes, or ``pose`` is set
        but it was built without move poses.
    """
    rec = {"action": PAST[change["action"]], "class": change["category"]}
    if volume:
        if "volume_m3" not in change:
            raise KeyError(NO_SIZES + " or set VLMCD.ANSWER_VOLUME False")
        if change["volume_m3"] is not None:
            rec["volume_m3"] = round_significant(change["volume_m3"])
    if has_distance(change):
        rec["distance_m"] = round(float(change["displacement_m"]), DISTANCE_DECIMALS)
        if pose != "none":
            if "rotation_deg" not in change:
                raise KeyError(NO_POSES)
            if change["rotation_deg"] is not None:
                rec["rotation_deg"] = change["rotation_deg"]
            if pose == "finalpose" and change["delta_xyz_m"] is not None:
                rec["final_pose"] = {
                    "translation_m": [round(float(x), POSE_DECIMALS) + 0.0 for x in change["delta_xyz_m"]],  # + 0.0: no -0.0
                    "yaw_deg": change["delta_yaw_deg"]}
    if bbox:
        if "aabb_extent_before" not in change:
            raise KeyError(NO_POSES)
        for state, acts in (("before", ("move", "remove")), ("after", ("move", "add"))):
            c, e = change[f"center_{state}"], change[f"aabb_extent_{state}"]
            if change["action"] in acts and change.get(f"seen_{state}", True) and c and e:
                rec[f"bbox_{state}"] = {"center": [round(float(x), POSE_DECIMALS) for x in c],
                                        "size": [round(float(x), POSE_DECIMALS) for x in e]}
    return rec


_EXPONENT_NUMBER = re.compile(r"(?<=[:,\[])(-?\d+(?:\.\d+)?[eE][-+]?\d+)")


def _json(obj):
    """Compact JSON (fewer tokens to generate), numbers always in positional notation.

    ``json.dumps`` writes floats below 1e-4 as ``1.4e-05``; small volumes would then mix
    notations with the rest, so those are rewritten as ``0.000014``.
    """
    text = json.dumps(obj, separators=(",", ":"))
    return _EXPONENT_NUMBER.sub(lambda m: format(Decimal(m.group(1)), "f"), text)


STATE_ENCODINGS = ("time", "offset")


def question(task, state_encoding="time", answer_format="json", order="size", volume=True,
             order_in_prompt=True, pose="none", bbox=False):
    """The prompt for a task.

    How the two scans reach the model depends on ``VLMCD.STATE_ENCODING`` (see
    ``VlmCdDatasetMapper``):

    ``time``
        two point clouds, each in its own ``<|vision_start|>...<|vision_end|>`` block, before
        first, both at the room's true coordinates and told apart by their time position.
    ``offset``
        one point cloud holding the room twice, the after scan shifted along +x.

    Parameters
    ----------
    task : str
        One of ``TASKS``.
    state_encoding : str
        One of ``STATE_ENCODINGS``.
    answer_format : str
        One of ``ANSWER_FORMATS``.
    order : str
        ``VLMCD.ANSWER_ORDER``; with ``size`` a ``json`` ``describe`` prompt states the order.
    volume : bool
        ``VLMCD.ANSWER_VOLUME``; a ``json`` ``describe`` prompt then asks for ``volume_m3``.
    order_in_prompt : bool
        ``VLMCD.ORDER_IN_PROMPT``; False leaves the order sentence out, which reproduces the
        prompt of models trained before it was added (JSON answers without volumes).
    pose : str
        ``VLMCD.ANSWER_POSE``; a ``json`` ``describe`` prompt then also asks for
        ``rotation_deg`` (``orient``) and ``final_pose`` (``finalpose``) of moved objects.
    bbox : bool
        ``VLMCD.ANSWER_BBOX``; a ``json`` ``describe`` prompt then also asks for the boxes.

    Returns
    -------
    str
    """
    if answer_format not in ANSWER_FORMATS:
        raise ValueError(f"unknown VLMCD.ANSWER_FORMAT {answer_format!r}; expected one of {ANSWER_FORMATS}")
    if state_encoding == "time":
        intro = ("The two point clouds show the same room scanned twice: the first before "
                 "and the second after some objects changed. ")
    elif state_encoding == "offset":
        intro = ("The point cloud shows the same room scanned twice: first before and then "
                 "after some objects changed; the after scan is shifted along the x axis. ")
    else:
        raise ValueError(f"unknown VLMCD.STATE_ENCODING {state_encoding!r}; expected one of {STATE_ENCODINGS}")
    if task == "describe" and answer_format == "json":
        fields = '{"action": "moved", "added" or "removed", "class": the object category, '
        if volume:
            fields += '"volume_m3": its bounding-box volume in cubic metres, '
        fields += '"distance_m": for a moved object, how far it moved in metres'
        if pose != "none":
            fields += (', "rotation_deg": for a moved object, how many degrees it turned '
                       '(0 to 180)')
        if pose == "finalpose":
            fields += (', "final_pose": for a moved object, its final pose relative to its '
                       'original pose: {"translation_m": [dx, dy, dz], the shift in metres '
                       'along the x, y and z axes of the point cloud, "yaw_deg": the turn '
                       'about the vertical z axis in degrees, counter-clockwise seen from above}')
        if bbox:
            fields += (', "bbox_before": for a moved or removed object, its axis-aligned bounding box '
                       'before the change, {"center": [x, y, z], "size": [dx, dy, dz]} in metres in '
                       'the coordinates of the point cloud, and "bbox_after": the same for a moved or '
                       'added object after the change')
        fields += '}.'
        text = ("List every object that was moved, added or removed as a JSON array "
                "with one entry per object: " + fields)
        if order == "size" and order_in_prompt:
            text += (" List the moved objects first, then the added, then the removed; "
                     "within each, the largest object first.")
        return intro + text
    if task == "describe":
        return intro + ("List every object that was moved, added or removed, "
                        "separated by semicolons.")
    if task == "count" and answer_format == "json":
        return intro + ("How many objects were moved, added and removed? Answer as JSON: "
                        '{"moved": n, "added": n, "removed": n}.')
    if task == "count":
        return intro + "How many objects were moved, added and removed?"
    raise ValueError(task)


ORDERS = ("size", "dataset")
NO_POSES = ("index.json has no move poses; rebuild it with `python "
            "datasets/dataset_preparation/vlm_cd/prepare_vlm_cd.py` or set VLMCD.ANSWER_POSE none")
NO_SIZES = ("index.json has no object sizes; rebuild it with `python vlm_cd/prepare_vlm_cd.py` "
            "(pyarrow >= 26, e.g. `sbatch vlm_cd/prepare_vlm_cd_slurm.sh`)")


def order_changes(changes, order="size"):
    """Put changes in the order the ``describe`` answer lists them.

    The language-modelling loss scores the answer token by token, so an arbitrary order
    of an unordered set adds loss the model cannot reduce. ``size`` makes the order a
    function of the scene: grouped by action (moved, added, removed, the same order as the
    ``count`` answer), then largest object first (object-frame bounding-box volume, so
    rotating an object does not change its rank), with the clause text breaking ties.

    Parameters
    ----------
    changes : list of dict
        Index change records.
    order : str
        ``"size"``, or ``"dataset"`` for the order of the dataset's ``changes`` column.

    Returns
    -------
    list of dict
    """
    if order == "dataset":
        return list(changes)
    if order != "size":
        raise ValueError(f"unknown VLMCD.ANSWER_ORDER {order!r}; expected one of {ORDERS}")
    if any("volume_m3" not in c for c in changes):
        raise KeyError(NO_SIZES + " or set VLMCD.ANSWER_ORDER dataset")
    # Identical objects (two chairs of one set) tie on volume; the clause text then decides.
    return sorted(changes, key=lambda c: (ACTIONS.index(c["action"]), -(c["volume_m3"] or 0.0),
                                          change_clause(c)))


def answer(task, changes, order="size", answer_format="json", volume=True, pose="none",
           bbox=False):
    """The target answer for a task.

    Parameters
    ----------
    task : str
        One of ``TASKS``.
    changes : list of dict
        Index change records, in dataset order.
    order : str
        Clause order of ``describe`` answers, see ``order_changes``.
    answer_format : str
        One of ``ANSWER_FORMATS``.
    volume : bool
        Give each ``json`` ``describe`` entry its ``volume_m3`` (``VLMCD.ANSWER_VOLUME``).
    pose : str
        ``VLMCD.ANSWER_POSE``, see ``change_record``.
    bbox : bool
        ``VLMCD.ANSWER_BBOX``, see ``change_record``.

    Returns
    -------
    str
    """
    if answer_format not in ANSWER_FORMATS:
        raise ValueError(f"unknown VLMCD.ANSWER_FORMAT {answer_format!r}; expected one of {ANSWER_FORMATS}")
    if answer_format == "json":
        if task == "describe":
            return _json([change_record(c, volume, pose, bbox) for c in order_changes(changes, order)])
        if task == "count":
            n = Counter(c["action"] for c in changes)
            return _json({PAST[a]: n[a] for a in ACTIONS})
        raise ValueError(task)
    if task == "describe":
        if not changes:
            return "no changes are visible"
        return "; ".join(change_clause(c) for c in order_changes(changes, order))
    if task == "count":
        n = Counter(c["action"] for c in changes)
        return ", ".join(f"{n[a]} {PAST[a]}" for a in ACTIONS)
    raise ValueError(task)


def _json_objects(text):
    """JSON objects in a model output, as dicts.

    Each top-level ``{...}`` (nested objects such as ``final_pose`` and the boxes stay
    inside their entry) is parsed on its own, so an array cut off by the token limit, or
    wrapped in extra text, still yields its complete entries. Pieces of a cut-off entry
    that parse alone (its nested objects) carry no ``action`` and are skipped by callers.

    Parameters
    ----------
    text : str

    Returns
    -------
    list of dict
    """
    out, dec, i = [], json.JSONDecoder(), 0
    while True:
        i = text.find("{", i)
        if i < 0:
            return out
        try:
            obj, j = dec.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        if isinstance(obj, dict):
            out.append(obj)
        i = j


def _number(obj, key):
    """``obj[key]`` as a finite float, or None when missing or not a number."""
    try:
        v = float(obj[key]) if obj.get(key) is not None else None
    except (TypeError, ValueError):
        return None
    return v if v is None or math.isfinite(v) else None


def _box(b):
    """``(cx, cy, cz, dx, dy, dz)`` of a ``{"center", "size"}`` box, or None when malformed."""
    try:
        vals = [float(x) for x in b["center"]] + [float(x) for x in b["size"]]
    except (TypeError, ValueError, KeyError):
        return None
    return tuple(vals) if len(vals) == 6 and all(math.isfinite(v) for v in vals) else None


def _pose(obj):
    """``(dx, dy, dz, yaw_deg)`` of ``obj["final_pose"]``, or None when missing or malformed."""
    fp = obj.get("final_pose")
    if not isinstance(fp, dict):
        return None
    t = fp.get("translation_m")
    try:
        vals = [float(x) for x in t] + [float(fp["yaw_deg"])]
    except (TypeError, ValueError, KeyError):
        return None
    return tuple(vals) if len(vals) == 4 and all(math.isfinite(v) for v in vals) else None


def parse_describe_entries(text):
    """Parse a ``describe`` answer, ``json`` or ``text``, into its entries in answer order.

    Parameters
    ----------
    text : str
        Model output or ground truth.

    Returns
    -------
    list of dict
        ``action`` (``move`` / ``add`` / ``remove``), ``category``, and ``distance_m``,
        ``volume_m3`` and ``rotation_deg`` as floats, or None when not given or not a number
        (``text`` answers never give a volume), and ``pose``: ``(dx, dy, dz, yaw_deg)`` of a
        ``final_pose``, or None, and ``bbox_before`` / ``bbox_after``: ``(cx, cy, cz, dx, dy, dz)``
        or None. Entries with an unknown action are skipped.
    """
    out = []
    if text.lstrip().startswith(("[", "{")):
        for obj in _json_objects(text):
            action = PRESENT.get(str(obj.get("action", "")).strip().lower())
            if action is None:
                continue
            out.append({"action": action, "category": clean_category(str(obj.get("class") or "")),
                        "distance_m": _number(obj, "distance_m"), "volume_m3": _number(obj, "volume_m3"),
                        "rotation_deg": _number(obj, "rotation_deg"), "pose": _pose(obj),
                        "bbox_before": _box(obj.get("bbox_before")),
                        "bbox_after": _box(obj.get("bbox_after"))})
        return out
    for clause in text.lower().split(";"):
        m = re.match(r"\s*(moved|added|removed)\s+(?:the|a|an)?\s*([a-z][a-z0-9]*)", clause)
        if m:
            d = re.search(r"\bby\s+(\d+(?:\.\d*)?)\s*m\b", clause)
            out.append({"action": PRESENT[m.group(1)], "category": m.group(2),
                        "distance_m": float(d.group(1)) if d else None, "volume_m3": None,
                        "rotation_deg": None, "pose": None, "bbox_before": None, "bbox_after": None})
    return out


def parse_describe_records(text):
    """Parse a ``describe`` answer, ``json`` or ``text``, into change records.

    Parameters
    ----------
    text : str
        Model output or ground truth.

    Returns
    -------
    list of tuple
        ``(action, category, distance_m)`` per entry of ``parse_describe_entries``.
    """
    return [(e["action"], e["category"], e["distance_m"]) for e in parse_describe_entries(text)]


def parse_describe(text):
    """Parse a ``describe`` answer back into ``(action, category)`` pairs.

    Parameters
    ----------
    text : str
        Model output or ground truth, ``json`` or ``text``.

    Returns
    -------
    collections.Counter
        Counts of ``(action, category)``.
    """
    return Counter((a, c) for a, c, _ in parse_describe_records(text))


def is_json_answer(text):
    """Whether ``text`` parses as a whole JSON value (no truncation, no extra text)."""
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def parse_count(text):
    """Parse a ``count`` answer.

    Parameters
    ----------
    text : str

    Returns
    -------
    dict
        ``{action: int or None}``.
    """
    objs = _json_objects(text) if text.lstrip().startswith("{") else []
    if objs:
        out = {}
        for a in ACTIONS:
            v = objs[0].get(PAST[a])
            out[a] = v if isinstance(v, int) and not isinstance(v, bool) else None
        return out
    out = {}
    for a in ACTIONS:
        m = re.search(rf"(\d+)\s+{PAST[a]}", text.lower())
        out[a] = int(m.group(1)) if m else None
    return out


def dataset_task(dataset_name):
    """Task of a registered dataset name ``vlmcd_<task>_<split>``.

    Parameters
    ----------
    dataset_name : str

    Returns
    -------
    str
    """
    return dataset_name.split("_")[1]


def load_vlmcd(task, split, root=None):
    """Dataset dicts of one registered dataset.

    Parameters
    ----------
    task : str
        One of ``TASKS``.
    split : str
        One of ``SPLITS``.
    root : str, optional
        Directory with the index and split files.

    Returns
    -------
    list of dict
        One ``{"example_id", "task", "split"}`` per example; the mapper looks the rest up
        in the index.
    """
    index = load_index(root)["examples"]
    ids = [i for i in load_split(split, root) if i in index]
    return [{"example_id": i, "task": task, "split": split, "image_id": i} for i in ids]


def register_vlmcd(root=None):
    """Register ``vlmcd_<task>_<split>`` for every task and split.

    The index is read lazily, when a dataset is first used, so registering is free even
    when the index has not been built.

    Parameters
    ----------
    root : str, optional
        Directory with the index and split files. Defaults to ``data_dir()`` at load time.
    """
    for task in TASKS:
        for split in SPLITS:
            name = f"vlmcd_{task}_{split}"
            if name in DatasetCatalog:
                continue
            DatasetCatalog.register(name, lambda t=task, s=split: load_vlmcd(t, s, root))
            MetadataCatalog.get(name).set(thing_classes=["object"], evaluator_type="vqa")
