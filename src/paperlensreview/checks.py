"""Pre-flight checks: NVIDIA GPU, ports, paperprep CLI presence."""
from __future__ import annotations

import logging
import shutil
import socket
import subprocess
from dataclasses import dataclass
from typing import Optional


log = logging.getLogger(__name__)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def check_nvidia_gpu() -> CheckResult:
    """Return one CheckResult for nvidia-smi presence + at least one visible GPU."""
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return CheckResult(
            "nvidia-gpu", False,
            "nvidia-smi not on PATH; paperlens-serve needs a CUDA GPU.",
        )
    try:
        out = subprocess.check_output(
            [nvsmi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            stderr=subprocess.STDOUT, text=True, timeout=10,
        )
    except subprocess.CalledProcessError as e:
        return CheckResult("nvidia-gpu", False, f"nvidia-smi exited {e.returncode}: {e.output.strip()}")
    except subprocess.TimeoutExpired:
        return CheckResult("nvidia-gpu", False, "nvidia-smi timeout")
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not lines:
        return CheckResult("nvidia-gpu", False, "nvidia-smi returned no GPUs")
    return CheckResult("nvidia-gpu", True, f"{len(lines)} GPU(s): {'; '.join(lines)}")


def check_port_free(host: str, port: int) -> CheckResult:
    """True if (host, port) isn't bound by another process."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        s.bind((host, port))
        s.close()
        return CheckResult(f"port-{port}-free", True, f"{host}:{port} available")
    except OSError as e:
        return CheckResult(f"port-{port}-free", False, f"{host}:{port} already in use: {e}")


def check_paperprep_cli(paperprep_module: str = "paperprep", python_bin: Optional[str] = None) -> CheckResult:
    """Verify `paperprep run --help` runs cleanly so subprocess invocations
    have a real chance of working.
    """
    cmd: list[str]
    if python_bin:
        cmd = [python_bin, "-m", paperprep_module, "run", "--help"]
    else:
        cmd = [paperprep_module, "run", "--help"]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=15)
    except FileNotFoundError:
        return CheckResult("paperprep-cli", False, f"binary not found: {cmd[0]}")
    except subprocess.CalledProcessError as e:
        return CheckResult("paperprep-cli", False, f"paperprep run --help failed: {e.output.strip()[:200]}")
    except subprocess.TimeoutExpired:
        return CheckResult("paperprep-cli", False, "paperprep run --help timed out")
    if "paperprep run" not in out and "Run the end-to-end pipeline" not in out:
        return CheckResult("paperprep-cli", False, f"paperprep --help output unexpected: {out[:200]}")
    return CheckResult("paperprep-cli", True, "paperprep CLI present")


def check_url_health(url: str, timeout: float = 3.0) -> CheckResult:
    """GET <url>/health and return ok if 200 + a JSON body."""
    import requests
    try:
        r = requests.get(url.rstrip("/") + "/health", timeout=timeout)
        if r.status_code == 200:
            try:
                return CheckResult(f"{url}-health", True, str(r.json()))
            except Exception:
                return CheckResult(f"{url}-health", True, r.text[:120])
        return CheckResult(f"{url}-health", False, f"HTTP {r.status_code}")
    except Exception as e:
        return CheckResult(f"{url}-health", False, f"{e}")
