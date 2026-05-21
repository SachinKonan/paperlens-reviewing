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

from .checks import check_paperprep_cli, check_url_health
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
    log.info("[paperlensreview] loaded config from %s", cfg_path)
    work_dir = Path(str(cfg.paperprep.work_dir)).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    _state["cfg"] = cfg
    _state["work_dir"] = work_dir
    _state["jobs"] = JobRegistry()
    # Serialize submissions (GPU-bound; MinerU + paperlens both contend)
    _state["job_lock"] = threading.Lock()
    log.info("[paperlensreview] work_dir=%s paperlens=%s", work_dir, cfg.paperlens_serve.base_url)


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
    ups = check_url_health(cfg.paperlens_serve.base_url)
    pp = check_paperprep_cli(cfg.paperprep.paperprep_module,
                             python_bin=(cfg.paperprep.python_bin or None))
    return {
        "status": "ok" if (ups.ok and pp.ok) else "degraded",
        "paperlens_serve": {"url": cfg.paperlens_serve.base_url, "healthy": ups.ok, "detail": ups.detail},
        "paperprep_cli":   {"healthy": pp.ok, "detail": pp.detail},
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
