"""Per-PDF pipeline: paperprep (subprocess) -> paperlens serve /score.

One job_id -> one subdir under ``cfg.paperprep.work_dir/<job_id>/``. Inside:

  uploaded.pdf                       raw upload
  manifest.jsonl                     paperprep input manifest (1 line: type=pdf)
  state.jsonl                        paperprep state log (one row per stage,paper)
  sharegpt/{text,vision}/data.json   paperprep export output
  pipeline.log                       captured stdout+stderr of paperprep subprocess
  result.json                        final verdict ({decision, p_accept, ...})

The UI polls ``status(job_id)`` which inspects state.jsonl + result.json to
compute the current stage and progress. Each stage runs to completion before
the next starts; we don't need a streaming protocol.
"""
from __future__ import annotations

import json
import logging
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests


log = logging.getLogger(__name__)


# Stage order = paperprep stages we run + the final paperlens scoring step.
# Compile is skipped for PDF input (paperprep auto-skips type=pdf entries).
STAGES: list[str] = ["mineru", "normalize", "filter", "export", "paperlens_score"]


@dataclass
class JobStatus:
    job_id: str
    state: str = "queued"               # queued | running | done | error
    stage: Optional[str] = None         # one of STAGES (or None before start / "done")
    stage_index: int = -1               # 0..len(STAGES)-1 (or -1 / len(STAGES))
    total_stages: int = len(STAGES)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    result: Optional[dict] = None       # final decision (decision, p_accept, ...)
    stage_log: list[dict] = field(default_factory=list)   # one entry per stage transition

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "state": self.state,
            "stage": self.stage,
            "stage_index": self.stage_index,
            "total_stages": self.total_stages,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result": self.result,
            "stage_log": list(self.stage_log),
        }


class JobRegistry:
    """In-memory store of JobStatus, keyed by job_id."""

    def __init__(self):
        self._jobs: dict[str, JobStatus] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str) -> JobStatus:
        with self._lock:
            s = JobStatus(job_id=job_id)
            self._jobs[job_id] = s
            return s

    def get(self, job_id: str) -> Optional[JobStatus]:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[JobStatus]:
        with self._lock:
            return list(self._jobs.values())


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _paperprep_cmd(cfg, job_dir: Path) -> list[str]:
    """Build the paperprep CLI invocation."""
    base: list[str]
    if cfg.paperprep.python_bin:
        base = [cfg.paperprep.python_bin, "-m", cfg.paperprep.paperprep_module, "run"]
    else:
        base = [cfg.paperprep.paperprep_module, "run"]
    args = [
        *base,
        "--input-manifest", str(job_dir / "manifest.jsonl"),
        "--output-dir", str(job_dir),
        "--stages", str(cfg.paperprep.stages),
        "--max-pages", str(int(cfg.paperprep.max_pages)),
        "--dpi", str(int(cfg.paperprep.dpi)),
        "--min-body-pages", str(int(cfg.paperprep.min_body_pages)),
    ]
    if cfg.paperprep.texlive_bin:
        args += ["--texlive-bin", str(cfg.paperprep.texlive_bin)]
    return args


def _walk_state_jsonl(state_path: Path) -> dict[str, str]:
    """Return {stage_name: latest_status} from paperprep's state.jsonl."""
    out: dict[str, str] = {}
    if not state_path.exists():
        return out
    with state_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            stage = row.get("stage")
            status = row.get("status")
            if stage and status:
                # Latest wins (resume-safe)
                out[stage] = status
    return out


def _stage_progress_watcher(job_dir: Path, status: JobStatus, stop_evt: threading.Event) -> None:
    """Background poller: while paperprep runs, watch state.jsonl and update
    ``status.stage`` / ``status.stage_index`` based on the latest stage with
    an "ok" row. Cheap (poll every 1s).
    """
    state_path = job_dir / "state.jsonl"
    paperprep_stages = STAGES[:-1]   # exclude paperlens_score (we drive that separately)
    last_seen: Optional[str] = None
    while not stop_evt.wait(1.0):
        states = _walk_state_jsonl(state_path)
        # Find the LATEST completed stage in our ordered list
        latest_ok = None
        for s in paperprep_stages:
            if states.get(s) == "ok":
                latest_ok = s
        # The "current" stage = next stage after latest_ok
        if latest_ok is None:
            cur = paperprep_stages[0]   # haven't completed anything yet -> running first
            cur_idx = 0
        else:
            i = paperprep_stages.index(latest_ok)
            cur_idx = min(i + 1, len(paperprep_stages) - 1)
            cur = paperprep_stages[cur_idx]
        if cur != last_seen:
            status.stage = cur
            status.stage_index = cur_idx
            status.stage_log.append({"t": time.time(), "stage": cur, "via": "paperprep_state"})
            last_seen = cur


def _write_manifest(pdf_path: Path, job_dir: Path, job_id: str) -> Path:
    """Write the 1-line paperprep input manifest pointing at the uploaded PDF."""
    mf = job_dir / "manifest.jsonl"
    mf.write_text(json.dumps({"id": job_id, "type": "pdf", "path": str(pdf_path)}) + "\n")
    return mf


def _load_sharegpt_export(job_dir: Path, modality: str) -> Optional[dict]:
    """Read the first sharegpt row from paperprep's export. Returns the row
    dict ready to POST to paperlens serve /score, or None if missing/empty.
    """
    p = job_dir / "sharegpt" / modality / "data.json"
    if not p.exists():
        log.warning(f"sharegpt export missing: {p}")
        return None
    try:
        rows = json.loads(p.read_text())
    except Exception as e:
        log.warning(f"sharegpt parse failed: {e}")
        return None
    if not rows:
        return None
    # paperprep emits inference-only rows ([system, human], no gpt). The
    # paperlens-serve /score endpoint goes through LF's "ppo" stage which
    # expects an assistant turn even at inference time. Append a neutral
    # placeholder; it's discarded after tokenization.
    row = dict(rows[0])
    convs = list(row.get("conversations", []))
    if not any(c.get("from") == "gpt" for c in convs):
        convs.append({"from": "gpt", "value": "Outcome: \\boxed{Accept}"})
    row["conversations"] = convs
    return row


def _score_via_paperlens(cfg, sharegpt_row: dict) -> dict:
    """POST one sharegpt row to paperlens serve /score, return its
    {p_accept, logp_accept, logp_reject, pred} dict.
    """
    url = cfg.paperlens_serve.base_url.rstrip("/") + "/score"
    payload = {"papers": [sharegpt_row]}
    r = requests.post(url, json=payload, timeout=float(cfg.paperlens_serve.timeout_seconds))
    r.raise_for_status()
    body = r.json()
    if not body.get("scores"):
        raise RuntimeError(f"paperlens-serve returned no scores: {body}")
    return body["scores"][0]


def _decision_from_p_accept(p_accept: float, threshold: float = 0.5) -> str:
    return "Accept" if p_accept >= threshold else "Reject"


def run_job(cfg, status: JobStatus, work_dir: Path, pdf_bytes: bytes, pdf_name: str) -> None:
    """End-to-end job. Writes ``result.json`` on success; sets ``status.error``
    on failure. Designed to be called inside a thread.
    """
    job_id = status.job_id
    job_dir = work_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Persist the uploaded PDF
    pdf_path = job_dir / f"{job_id}.pdf"
    pdf_path.write_bytes(pdf_bytes)
    log.info(f"[{job_id}] pdf saved -> {pdf_path} ({len(pdf_bytes)} bytes)")

    _write_manifest(pdf_path, job_dir, job_id)
    status.state = "running"
    status.started_at = time.time()
    status.stage = STAGES[0]
    status.stage_index = 0

    # Watcher thread updates status.stage/stage_index as paperprep emits state.jsonl rows
    stop_evt = threading.Event()
    watcher = threading.Thread(target=_stage_progress_watcher,
                               args=(job_dir, status, stop_evt), daemon=True)
    watcher.start()

    try:
        # ---- run paperprep ----
        cmd = _paperprep_cmd(cfg, job_dir)
        log.info(f"[{job_id}] paperprep: {shlex.join(cmd)}")
        with (job_dir / "pipeline.log").open("w") as lf:
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                               text=True, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"paperprep exited {r.returncode}; see pipeline.log")

        states = _walk_state_jsonl(job_dir / "state.jsonl")
        if states.get("export") != "ok":
            failed_stages = {s: st for s, st in states.items() if st != "ok"}
            raise RuntimeError(
                f"paperprep did not reach export=ok; latest states: {failed_stages or states}"
            )

        # ---- paperlens score ----
        status.stage = "paperlens_score"
        status.stage_index = len(STAGES) - 1
        status.stage_log.append({"t": time.time(), "stage": "paperlens_score", "via": "transition"})

        modality = str(cfg.review.modality)
        row = _load_sharegpt_export(job_dir, modality)
        if row is None:
            raise RuntimeError(f"paperprep export produced no {modality} sharegpt row")
        log.info(f"[{job_id}] scoring via paperlens-serve {cfg.paperlens_serve.base_url} (modality={modality})")
        score = _score_via_paperlens(cfg, row)

        result = {
            "decision": _decision_from_p_accept(float(score["p_accept"])),
            "p_accept": float(score["p_accept"]),
            "logp_accept": score.get("logp_accept"),
            "logp_reject": score.get("logp_reject"),
            "pred": score.get("pred"),
            "modality": modality,
            "domain": str(cfg.review.domain),
            "pdf_name": pdf_name,
            "job_dir": str(job_dir),
        }
        (job_dir / "result.json").write_text(json.dumps(result, indent=2))
        status.result = result
        status.state = "done"
        status.stage = "done"
        status.stage_index = len(STAGES)
        log.info(f"[{job_id}] result: {result['decision']} (p_accept={result['p_accept']:.4f})")
    except Exception as e:
        log.exception(f"[{job_id}] pipeline failed")
        status.state = "error"
        status.error = str(e)
    finally:
        stop_evt.set()
        watcher.join(timeout=2)
        status.finished_at = time.time()
