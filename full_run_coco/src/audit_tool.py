"""ipywidgets review tool for the label audit (Stage A, and Stage B with error tags).

Use from a notebook running the cmpe297 kernel (notebooks/label_audit.ipynb):
    import sys; sys.path.insert(0, "../src")
    from audit_tool import AuditTool, stage_b
    AuditTool("data/audit/audit_sample_coco.csv", reviewer="alice").show()   # Stage A
    stage_b("fp", "coco", reviewer="alice").show()                           # Stage B false positives

Blind review: the reviewer sees only the image and the object name, never the dataset label,
the split, or any model answer, and no annotation boxes (drawing a box would reveal that the
label is "yes"). Stage B lists contain only model errors, so there the reviewer knows the
dataset label (unavoidable); the verdict still records what is actually in the image.

Each click writes the whole labels CSV (default data/audit/labels_<dataset>_<reviewer>.csv),
so the tool can be closed at any time and resumes at the first unlabelled item. Back / Next
revisit items; clicking a verdict again replaces the earlier one.

Reviewer guidelines (also stated in the report's methods section):
- Present: any clearly recognisable instance, including small, partial, or depicted ones
  (in a poster, on a screen, in a mirror); tick "depicted" for those.
- Absent: confident it is not there after inspecting the whole image (use the zoom).
- Uncertain: too small, blurry or ambiguous to decide, or the category boundary is unclear
  (for example bench vs chair, cup vs mug).
"""
import csv
import io
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import ipywidgets as w
from PIL import Image

from pope_sampling import image_path

PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = os.environ.get("CMPE297_DATA", os.path.expanduser("~/cmpe297/data"))
VERDICTS = ["present", "absent", "uncertain"]
LABEL_COLUMNS = ["audit_id", "dataset", "image", "object", "reviewer", "verdict", "depicted",
                 "tag", "note", "labelled_utc"]
# Stage B error taxonomy. "auto" maps the verdict that means the dataset label was wrong to
# the tag filled in when the reviewer picks no tag (they may still choose another, e.g. a
# depicted object counts as present but is tagged "depicted or reflected").
STAGE_B = {
    "fp": {"tags": ["label error", "co-occurrence prior", "similar object", "depicted or reflected",
                    "ambiguous category", "other / no visible cause"],
           "auto": {"present": "label error"}},
    "fn": {"tags": ["label error", "small", "occluded or truncated", "unusual appearance", "other"],
           "auto": {"absent": "label error"}},
}
GUIDE = ("<b>Present</b>: any clearly recognisable instance, including small, partial, or depicted "
         "(poster, screen, mirror; tick <i>depicted</i>). <b>Absent</b>: confident after inspecting "
         "the whole image. <b>Uncertain</b>: too small, blurry or ambiguous, or unclear category "
         "boundary (bench vs chair).")


def read_csv(path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def default_labels_path(sample_csv, reviewer: str) -> Path:
    """audit_sample_coco.csv -> labels_coco_<reviewer>.csv; audit_fp_coco.csv -> labels_fp_coco_<reviewer>.csv"""
    stem = Path(sample_csv).stem.removeprefix("audit_sample_").removeprefix("audit_")
    return PROJECT / "data/audit" / f"labels_{stem}_{reviewer}.csv"


def stage_b(kind: str, dataset: str, reviewer: str) -> "AuditTool":
    """Review tool for the Stage B list audit_<kind>_<dataset>.csv, kind "fp" or "fn"."""
    return AuditTool(f"data/audit/audit_{kind}_{dataset}.csv", reviewer,
                     tags=STAGE_B[kind]["tags"], auto_tags=STAGE_B[kind]["auto"])


class AuditTool:
    def __init__(self, sample_csv, reviewer: str, labels_csv=None,
                 tags=None, auto_tags=None, data_root=DATA_ROOT):
        if not re.fullmatch(r"[a-z0-9_]+", reviewer):
            raise ValueError("reviewer must be lower-case letters, digits or _ (it goes in the file name)")
        sample_csv = PROJECT / sample_csv if not Path(sample_csv).is_absolute() else Path(sample_csv)
        self.items = read_csv(sample_csv)
        self.reviewer, self.tags, self.auto_tags, self.data_root = reviewer, tags, auto_tags or {}, data_root
        self.labels_csv = Path(labels_csv) if labels_csv else default_labels_path(sample_csv, reviewer)
        self.labels = {r["audit_id"]: r for r in read_csv(self.labels_csv)} if self.labels_csv.exists() else {}
        self.pos = self._first_unlabelled()
        self._build()

    # ---- state -------------------------------------------------------------------------
    def _first_unlabelled(self) -> int:
        return next((i for i, it in enumerate(self.items) if it["audit_id"] not in self.labels),
                    len(self.items))

    def progress(self) -> str:
        done = sum(it["audit_id"] in self.labels for it in self.items)
        return f"{done}/{len(self.items)} labelled"

    def record(self, verdict: str, depicted=False, tag="", note=""):
        """Store a verdict for the current item and save the labels file."""
        if verdict not in VERDICTS:
            raise ValueError(verdict)
        tag = tag or self.auto_tags.get(verdict, "")
        if self.tags and not tag:
            raise ValueError("choose a tag first")
        it = self.items[self.pos]
        self.labels[it["audit_id"]] = {
            "audit_id": it["audit_id"], "dataset": it["dataset"], "image": it["image"],
            "object": it["object"], "reviewer": self.reviewer, "verdict": verdict,
            "depicted": int(bool(depicted)), "tag": tag, "note": note.strip(),
            "labelled_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._save()

    def _save(self):
        order = {it["audit_id"]: i for i, it in enumerate(self.items)}
        rows = sorted(self.labels.values(), key=lambda r: order.get(r["audit_id"], len(order)))
        self.labels_csv.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.labels_csv.with_suffix(".tmp")
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, LABEL_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, self.labels_csv)

    # ---- widgets -----------------------------------------------------------------------
    def _build(self):
        self.header = w.HTML()
        self.image = w.Image(format="jpeg")
        self.zoom = w.ToggleButtons(options=[("1x", 1.0), ("1.5x", 1.5), ("2x", 2.0)], value=1.0,
                                    description="Zoom")
        self.depicted = w.Checkbox(description="depicted (poster / screen / mirror / toy)", indent=False)
        self.note = w.Text(placeholder="optional note", layout=w.Layout(width="60%"))
        self.tag = w.Dropdown(options=[""] + list(self.tags or []), description="Tag")
        buttons = []
        for verdict, style in zip(VERDICTS, ["success", "danger", "warning"]):
            b = w.Button(description=verdict.capitalize(), button_style=style)
            b.on_click(lambda _, v=verdict: self._click(v))
            buttons.append(b)
        back, nxt = w.Button(description="< Back"), w.Button(description="Next >")
        back.on_click(lambda _: self._go(self.pos - 1))
        nxt.on_click(lambda _: self._go(self.pos + 1))
        self.status = w.HTML()
        self.zoom.observe(lambda _: self._render(), names="value")
        controls = [w.HBox(buttons + [back, nxt]), w.HBox([self.depicted, self.note])]
        if self.tags:
            controls.insert(1, self.tag)
        self.box = w.VBox([w.HTML(GUIDE), self.header, self.zoom, self.image, *controls, self.status])
        self._render()

    def _click(self, verdict):
        try:
            self.record(verdict, self.depicted.value, self.tag.value, self.note.value)
        except ValueError as e:
            self.status.value = f"<span style='color:red'>{e}</span>"
            return
        self._go(self.pos + 1)

    def _go(self, pos):
        self.pos = max(0, min(pos, len(self.items)))
        self._render()

    def _render(self):
        self.status.value = self.progress() + f" &nbsp; saving to <code>{self.labels_csv.name}</code>"
        if self.pos >= len(self.items):
            left = len(self.items) - sum(it["audit_id"] in self.labels for it in self.items)
            self.header.value = ("<h3>End of list. " + (f"{left} item(s) still unlabelled; rerun the cell to "
                                 "jump to the first one." if left else "All items labelled.")
                                 + " Use Back to revisit any item.</h3>")
            self.image.value = b""
            return
        it = self.items[self.pos]
        prev = self.labels.get(it["audit_id"])
        self.header.value = (f"<h3>{self.pos + 1}/{len(self.items)} &nbsp; {it['audit_id']}: "
                             f"is there <u>{it['object']}</u> in the image?</h3>"
                             + (f"<i>current verdict: {prev['verdict']}</i>" if prev else ""))
        self.depicted.value = bool(prev) and str(prev["depicted"]) == "1"
        self.note.value = prev["note"] if prev else ""
        self.tag.value = prev["tag"] if prev and prev["tag"] in self.tag.options else ""
        img = Image.open(image_path(self.data_root, it["dataset"], it["image"])).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        self.image.value = buf.getvalue()
        self.image.layout.width = f"{int(img.width * self.zoom.value)}px"

    def show(self):
        return self.box
