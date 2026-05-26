"""FastAPI server for paperlensreview.

Endpoints:
  POST /submit              upload a PDF -> {job_id}
  GET  /status/{job_id}     poll for progress + result
  GET  /health              service + upstream healthcheck
  GET  /                    minimal HTML UI

The pipeline runs in a background thread (one job at a time per worker --
MinerU + paperlens both want GPU). For higher throughput, run multiple
paperlensreview workers behind a queue.
"""
from __future__ import annotations

import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from omegaconf import OmegaConf

from .checks import check_url_health
from .pipeline import JobRegistry, JobStatus, STAGES, run_job


log = logging.getLogger(__name__)


# --------- App + state ---------

_state: dict[str, Any] = {}
app = FastAPI(title="paperlens-reviewing")

_UI_DIR = Path(__file__).resolve().parent / "ui"


@app.on_event("startup")
def _startup() -> None:
    cfg_path = os.environ.get("PAPERLENSREVIEW_CONFIG", "configs/server.yaml")
    cfg = OmegaConf.load(cfg_path)
    if (env := os.environ.get("PAPERLENS_SERVE_BASE_URL")):
        cfg.paperlens_serve.base_url = env
    if (env := os.environ.get("PAPERPREP_SERVE_BASE_URL")):
        cfg.paperprep_serve.base_url = env
    log.info("[paperlensreview] loaded config from %s", cfg_path)
    work_dir = Path(str(cfg.paperprep_serve.work_dir)).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    _state["cfg"] = cfg
    _state["work_dir"] = work_dir
    _state["jobs"] = JobRegistry()
    # Serialize submissions: paperprep serve + paperlens serve each hold a vLLM
    # engine on the same GPU; interleaved /prepare + /score calls thrash.
    _state["job_lock"] = threading.Lock()
    log.info("[paperlensreview] work_dir=%s paperlens=%s paperprep=%s",
             work_dir, cfg.paperlens_serve.base_url, cfg.paperprep_serve.base_url)


# --------- Endpoints ---------

@app.get("/")
def root() -> FileResponse:
    idx = _UI_DIR / "index.html"
    if not idx.exists():
        raise HTTPException(500, f"UI not bundled: {idx}")
    return FileResponse(idx)


@app.get("/health")
def health() -> dict:
    cfg = _state["cfg"]
    pl = check_url_health(cfg.paperlens_serve.base_url, path="/health")
    pp = check_url_health(cfg.paperprep_serve.base_url, path="/healthz")
    return {
        "status": "ok" if (pl.ok and pp.ok) else "degraded",
        "paperlens_serve": {"url": cfg.paperlens_serve.base_url, "healthy": pl.ok, "detail": pl.detail},
        "paperprep_serve": {"url": cfg.paperprep_serve.base_url, "healthy": pp.ok, "detail": pp.detail},
        "n_jobs": len(_state["jobs"].all()),
        "stages": STAGES,
    }


@app.post("/submit")
def submit(file: UploadFile = File(...)) -> dict:
    """Accept a PDF upload, kick off the pipeline in a background thread,
    return the job_id immediately. The UI then polls /status/{job_id}.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "only .pdf uploads are supported")
    pdf_bytes = file.file.read()
    if not pdf_bytes:
        raise HTTPException(400, "empty file")
    if len(pdf_bytes) > 100 * 1024 * 1024:
        raise HTTPException(413, f"PDF too large: {len(pdf_bytes)} bytes (>100 MB cap)")

    job_id = uuid.uuid4().hex[:12]
    status = _state["jobs"].create(job_id)
    log.info("[/submit] job=%s file=%r size=%d", job_id, file.filename, len(pdf_bytes))

    def _runner():
        # One pipeline at a time per worker process (MinerU + paperlens are
        # GPU-bound; they thrash if interleaved).
        with _state["job_lock"]:
            run_job(_state["cfg"], status, _state["work_dir"], pdf_bytes, file.filename)

    t = threading.Thread(target=_runner, daemon=True, name=f"job-{job_id}")
    t.start()
    return {"job_id": job_id, "stages": STAGES, "submitted_at": status.started_at}


@app.get("/status/{job_id}")
def status_endpoint(job_id: str) -> dict:
    j = _state["jobs"].get(job_id)
    if not j:
        raise HTTPException(404, f"unknown job_id: {job_id!r}")
    return j.to_dict()


@app.get("/jobs")
def list_jobs() -> list[dict]:
    return [j.to_dict() for j in _state["jobs"].all()]


# --------- Per-job file explorer (read-only) ---------

import json as _json
import mimetypes
from pathlib import Path as _Path


_TREE_MAX_FILES_PER_DIR = 500


def _job_roots(job_id: str) -> dict[str, _Path]:
    """Return {label: root_path} pairs the explorer is allowed to read."""
    j = _state["jobs"].get(job_id)
    if not j:
        raise HTTPException(404, f"unknown job_id: {job_id!r}")
    if not j.result:
        raise HTTPException(409, f"job {job_id} not done yet (state={j.state!r})")
    roots: dict[str, _Path] = {"review": _Path(j.result["job_dir"]).resolve()}
    pp = j.result.get("paperprep_output_dir")
    if pp:
        roots["paperprep"] = _Path(pp).resolve()
    return roots


def _list_tree(root: _Path) -> dict:
    if not root.exists():
        return {"name": root.name, "path": str(root), "type": "missing"}
    if root.is_file():
        return {"name": root.name, "path": str(root), "type": "file",
                "size": root.stat().st_size}
    children: list[dict] = []
    try:
        entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        return {"name": root.name, "path": str(root), "type": "dir",
                "children": [], "error": "permission denied"}
    for child in entries:
        if len(children) >= _TREE_MAX_FILES_PER_DIR:
            children.append({"name": "...", "type": "truncated"})
            break
        children.append(_list_tree(child))
    return {"name": root.name, "path": str(root), "type": "dir", "children": children}


@app.get("/jobs/{job_id}/tree")
def job_tree(job_id: str) -> dict:
    return {label: _list_tree(root) for label, root in _job_roots(job_id).items()}


@app.get("/jobs/{job_id}/file")
def job_file(job_id: str, path: str):
    """Stream a file from the job's allowed roots. Rejects path-traversal."""
    target = _Path(path).resolve()
    roots = _job_roots(job_id)
    if not any(target == r or target.is_relative_to(r) for r in roots.values()):
        raise HTTPException(403, f"path not under any allowed root for job {job_id}")
    if not target.is_file():
        raise HTTPException(404, f"not a file: {target}")
    media_type, _ = mimetypes.guess_type(str(target))
    if media_type is None:
        # Default to text/plain for unknown extensions so the browser inlines them
        media_type = "text/plain; charset=utf-8"
    headers = {"Content-Disposition": f'inline; filename="{target.name}"'}
    return FileResponse(target, media_type=media_type, headers=headers)


@app.get("/jobs/{job_id}/payload")
def job_payload(job_id: str) -> dict:
    """Return the exact sharegpt row paperlens-serve scored, plus the score."""
    j = _state["jobs"].get(job_id)
    if not j:
        raise HTTPException(404, f"unknown job_id: {job_id!r}")
    if not j.result:
        raise HTTPException(409, f"job {job_id} not done yet")
    modality = j.result.get("modality", "vision")
    pp = j.result.get("paperprep_output_dir")
    if not pp:
        raise HTTPException(404, "paperprep_output_dir missing from result")
    sg_path = _Path(pp) / "sharegpt" / modality / "data.json"
    if not sg_path.exists():
        raise HTTPException(404, f"sharegpt export not found: {sg_path}")
    rows = _json.loads(sg_path.read_text())
    row = rows[0] if rows else None
    if row is not None:
        # Mirror what pipeline.py appends before POSTing to paperlens-serve.
        convs = list(row.get("conversations", []))
        if not any(c.get("from") == "gpt" for c in convs):
            convs.append({"from": "gpt", "value": "Outcome: \\boxed{Accept}"})
            row = {**row, "conversations": convs}
    return {
        "job_id": job_id,
        "modality": modality,
        "sharegpt_path": str(sg_path),
        "row": row,
        "score": j.result,
    }


# --------- Entry point ---------

def run_uvicorn(cfg_path: str, host: Optional[str] = None, port: Optional[int] = None) -> int:
    import uvicorn
    cfg = OmegaConf.load(cfg_path)
    host = host or cfg.server.host
    port = port or int(cfg.server.port)
    os.environ["PAPERLENSREVIEW_CONFIG"] = cfg_path
    log_level = str(cfg.get("logging", {}).get("level", "INFO")).lower()
    logging.basicConfig(level=log_level.upper(),
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    log.info("paperlensreview binding %s:%s", host, port)
    uvicorn.run(app, host=host, port=port, log_level=log_level)
    return 0
