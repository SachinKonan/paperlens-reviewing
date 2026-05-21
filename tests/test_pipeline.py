"""Pipeline / server unit tests. No GPU, no paperprep subprocess --
we mock the heavy steps so the wiring stays under test.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from paperlensreview.pipeline import (
    STAGES, JobRegistry, JobStatus, _decision_from_p_accept,
    _load_sharegpt_export, _walk_state_jsonl, run_job,
)


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------

def test_decision_threshold():
    assert _decision_from_p_accept(0.51) == "Accept"
    assert _decision_from_p_accept(0.49) == "Reject"
    assert _decision_from_p_accept(0.5)  == "Accept"  # tie goes to accept


def test_state_jsonl_latest_wins(tmp_path: Path):
    p = tmp_path / "state.jsonl"
    p.write_text(
        json.dumps({"id": "a", "stage": "mineru",    "status": "failed"}) + "\n" +
        json.dumps({"id": "a", "stage": "mineru",    "status": "ok"}) + "\n" +
        json.dumps({"id": "a", "stage": "normalize", "status": "ok"}) + "\n"
    )
    assert _walk_state_jsonl(p) == {"mineru": "ok", "normalize": "ok"}


def test_sharegpt_export_appends_gpt_turn(tmp_path: Path):
    """paperprep emits [system, human] only. _load_sharegpt_export must
    append a placeholder gpt turn so LF's 'ppo' stage doesn't drop the row."""
    ds_dir = tmp_path / "sharegpt" / "vision"
    ds_dir.mkdir(parents=True)
    rows = [{
        "conversations": [
            {"from": "system", "value": "sys"},
            {"from": "human",  "value": "user"},
        ],
        "_metadata": {"id": "x"},
        "images":    ["/x/page_1.png"],
    }]
    (ds_dir / "data.json").write_text(json.dumps(rows))

    out = _load_sharegpt_export(tmp_path, "vision")
    assert out is not None
    roles = [c["from"] for c in out["conversations"]]
    assert roles == ["system", "human", "gpt"]
    # The gpt turn should not overwrite an existing one when present
    rows[0]["conversations"].append({"from": "gpt", "value": "real"})
    (ds_dir / "data.json").write_text(json.dumps(rows))
    out2 = _load_sharegpt_export(tmp_path, "vision")
    assert out2["conversations"][-1] == {"from": "gpt", "value": "real"}


def test_registry_create_get():
    r = JobRegistry()
    s = r.create("abc")
    assert isinstance(s, JobStatus)
    assert s.job_id == "abc"
    assert r.get("abc") is s
    assert r.get("missing") is None
    assert len(r.all()) == 1


# ---------------------------------------------------------------------------
# Integration: run_job with mocked subprocess + mocked paperlens-serve
# ---------------------------------------------------------------------------

def _make_cfg(tmp_path: Path) -> OmegaConf:
    return OmegaConf.create({
        "server":          {"host": "127.0.0.1", "port": 0},
        "paperlens_serve": {"base_url": "http://x.y", "timeout_seconds": 5},
        "paperprep":       {
            "python_bin": "",
            "paperprep_module": "paperprep",
            "work_dir": str(tmp_path / "work"),
            "stages": "mineru,normalize,filter,export",
            "max_pages": 14, "dpi": 150, "min_body_pages": 6,
            "texlive_bin": "",
        },
        "review":          {"modality": "vision", "domain": "arxiv"},
    })


def test_run_job_happy_path(tmp_path: Path):
    cfg = _make_cfg(tmp_path)
    work = Path(cfg.paperprep.work_dir); work.mkdir(parents=True)

    # Fake subprocess.run that writes a state.jsonl with all-ok rows + a
    # sharegpt export, returning rc=0.
    def fake_run(cmd, stdout=None, stderr=None, text=None, check=False):
        # cmd = [..., "--output-dir", <job_dir>, ...]
        out_dir_idx = cmd.index("--output-dir") + 1
        job_dir = Path(cmd[out_dir_idx])
        (job_dir / "state.jsonl").write_text("\n".join(
            json.dumps({"id": "j", "stage": s, "status": "ok"})
            for s in ["mineru", "normalize", "filter", "export"]
        ))
        sg = job_dir / "sharegpt" / "vision"
        sg.mkdir(parents=True, exist_ok=True)
        (sg / "data.json").write_text(json.dumps([{
            "conversations": [
                {"from": "system", "value": "sys"},
                {"from": "human",  "value": "user"},
            ],
            "_metadata": {"id": "j"},
            "images":    [],
        }]))
        class _R: returncode = 0
        return _R()

    # Fake requests.post returning a p_accept
    def fake_post(url, json=None, timeout=None):
        class _R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return {"scores": [{"p_accept": 0.77, "logp_accept": -0.3, "logp_reject": -1.5, "pred": "Outcome: \\boxed{Accept}"}]}
        return _R()

    status = JobStatus(job_id="j")
    with patch("paperlensreview.pipeline.subprocess.run", side_effect=fake_run), \
         patch("paperlensreview.pipeline.requests.post", side_effect=fake_post):
        run_job(cfg, status, work, b"%PDF-1.4 fake", "fake.pdf")

    assert status.state == "done", status.error
    assert status.stage == "done"
    assert status.stage_index == len(STAGES)
    assert status.result is not None
    assert status.result["decision"] == "Accept"
    assert abs(status.result["p_accept"] - 0.77) < 1e-9


def test_run_job_paperprep_failure(tmp_path: Path):
    cfg = _make_cfg(tmp_path)
    work = Path(cfg.paperprep.work_dir); work.mkdir(parents=True)

    def fake_run(cmd, stdout=None, stderr=None, text=None, check=False):
        class _R: returncode = 2
        return _R()

    status = JobStatus(job_id="boom")
    with patch("paperlensreview.pipeline.subprocess.run", side_effect=fake_run):
        run_job(cfg, status, work, b"%PDF-1.4 fake", "fake.pdf")
    assert status.state == "error"
    assert "paperprep exited 2" in (status.error or "")


def test_run_job_export_missing(tmp_path: Path):
    """paperprep returns 0 but no export row -> mark error."""
    cfg = _make_cfg(tmp_path)
    work = Path(cfg.paperprep.work_dir); work.mkdir(parents=True)

    def fake_run(cmd, **kwargs):
        out_dir_idx = cmd.index("--output-dir") + 1
        job_dir = Path(cmd[out_dir_idx])
        # only mineru completed
        (job_dir / "state.jsonl").write_text(
            json.dumps({"id": "j", "stage": "mineru", "status": "ok"}) + "\n"
        )
        class _R: returncode = 0
        return _R()

    status = JobStatus(job_id="j")
    with patch("paperlensreview.pipeline.subprocess.run", side_effect=fake_run):
        run_job(cfg, status, work, b"%PDF-1.4 fake", "fake.pdf")
    assert status.state == "error"
    assert "export=ok" in (status.error or "")
