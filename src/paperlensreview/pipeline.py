"""Per-PDF pipeline: paperprep serve /prepare -> paperlens serve /score.

One job_id -> one subdir under ``cfg.paperprep_serve.work_dir/<job_id>/``. Inside:

  <job_id>.pdf       raw upload (path we hand to paperprep over the wire)
  result.json        final verdict ({decision, p_accept, ...})

The paperprep serve daemon owns its own output_dir (per-request subdirs hold
the compile/mineru/normalize/filter/export artifacts). We don't touch those;
we read the absolute ``sharegpt_vision_path`` / ``sharegpt_text_path`` it
returns and POST a single row to paperlens serve /score.

Both upstream services are persistent FastAPI/Flask daemons -- launch_local.sh
brings them up and waits for /healthz before exec'ing the reviewing server.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests


log = logging.getLogger(__name__)


# Two coarse stages: paperprep serve runs synchronously and only returns when
# its internal compile/mineru/normalize/filter/export chain is done, so we
# can't show fine-grained progress for it without modifying paperprep itself.
STAGES: list[str] = ["paperprep", "paperlens_score"]


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
# Stage transition helper
# ---------------------------------------------------------------------------

def _enter_stage(status: JobStatus, stage: str, via: str = "transition") -> None:
    status.stage = stage
    try:
        status.stage_index = STAGES.index(stage)
    except ValueError:
        status.stage_index = len(STAGES)
    status.stage_log.append({"t": time.time(), "stage": stage, "via": via})


# ---------------------------------------------------------------------------
# paperprep serve client
# ---------------------------------------------------------------------------

def _call_paperprep_prepare(cfg, job_id: str, pdf_path: Path) -> dict:
    """POST one PDF to paperprep serve /prepare. Returns the response dict.

    paperprep serve writes its outputs under its own output_dir (configured at
    daemon launch). The response has absolute paths to the ShareGPT exports we
    then forward to paperlens-serve.
    """
    url = cfg.paperprep_serve.base_url.rstrip("/") + "/prepare"
    payload = {
        "request_id": job_id,
        "papers": [
            {"id": job_id, "type": "pdf", "path": str(pdf_path)},
        ],
    }
    timeout = float(cfg.paperprep_serve.timeout_seconds)
    r = requests.post(url, json=payload, timeout=timeout)
    if r.status_code >= 500:
        raise RuntimeError(f"paperprep serve {r.status_code}: {r.text[:400]}")
    r.raise_for_status()
    body = r.json()
    if not isinstance(body, dict):
        raise RuntimeError(f"paperprep serve returned non-dict body: {body!r}")
    return body


def _load_sharegpt_export(prepare_body: dict, modality: str) -> Optional[dict]:
    """Resolve which sharegpt_{modality}_path to read from paperprep's response,
    load the first row, and append a placeholder gpt turn so LF's 'ppo' stage
    accepts it at inference time. Returns None on any miss.
    """
    key = f"sharegpt_{modality}_path"
    sg_path_str = prepare_body.get(key)
    if not sg_path_str:
        log.warning("paperprep prepare body missing %s: %r", key, prepare_body)
        return None
    sg_path = Path(sg_path_str)
    if not sg_path.exists():
        log.warning("paperprep sharegpt export path doesn't exist: %s", sg_path)
        return None
    try:
        rows = json.loads(sg_path.read_text())
    except Exception as e:
        log.warning("paperprep sharegpt parse failed (%s): %s", sg_path, e)
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


# ---------------------------------------------------------------------------
# paperlens serve client
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main job entry point
# ---------------------------------------------------------------------------

def run_job(cfg, status: JobStatus, work_dir: Path, pdf_bytes: bytes, pdf_name: str) -> None:
    """End-to-end job. Writes ``result.json`` on success; sets ``status.error``
    on failure. Designed to be called inside a thread.
    """
    job_id = status.job_id
    job_dir = work_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = job_dir / f"{job_id}.pdf"
    pdf_path.write_bytes(pdf_bytes)
    log.info(f"[{job_id}] pdf saved -> {pdf_path} ({len(pdf_bytes)} bytes)")

    status.state = "running"
    status.started_at = time.time()
    _enter_stage(status, "paperprep")

    try:
        log.info(f"[{job_id}] paperprep serve /prepare ({cfg.paperprep_serve.base_url}, pdf={pdf_path})")
        prepare_body = _call_paperprep_prepare(cfg, job_id, pdf_path)

        # Validate the per-paper result before claiming success
        papers = prepare_body.get("papers") or []
        if not papers:
            raise RuntimeError(f"paperprep prepare returned no papers: {prepare_body}")
        p0 = papers[0]
        if p0.get("status") != "ok":
            raise RuntimeError(
                f"paperprep failed at paper level: status={p0.get('status')!r} "
                f"error={p0.get('error')!r}"
            )

        _enter_stage(status, "paperlens_score")
        modality = str(cfg.review.modality)
        row = _load_sharegpt_export(prepare_body, modality)
        if row is None:
            raise RuntimeError(
                f"paperprep prepare did not produce a {modality} sharegpt row; "
                f"response keys={list(prepare_body.keys())}"
            )
        log.info(f"[{job_id}] paperlens serve /score ({cfg.paperlens_serve.base_url}, modality={modality})")
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
            "paperprep_output_dir": prepare_body.get("output_dir"),
            "paperprep_elapsed_s": prepare_body.get("elapsed_s"),
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
        status.finished_at = time.time()
