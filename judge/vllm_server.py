"""judge/vllm_server.py — vLLM OpenAI-compatible server lifecycle."""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class VllmServerError(RuntimeError):
    pass


class VllmServer:
    def __init__(
        self,
        model_id: str,
        port: int,
        *,
        max_model_len: int = 8192,
        log_dir: Path,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "auto",
        quantization: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.port = port
        self.max_model_len = max_model_len
        self.log_dir = Path(log_dir)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.quantization = quantization
        self._proc: Optional[subprocess.Popen] = None  # type: ignore[type-arg]

    def start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = self.log_dir / f"vllm_{self.port}_stdout.log"
        stderr_log = self.log_dir / f"vllm_{self.port}_stderr.log"
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_id,
            "--port", str(self.port),
            "--host", "127.0.0.1",
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--dtype", self.dtype,
            "--disable-log-requests",
        ]
        if self.quantization:
            cmd += ["--quantization", self.quantization]
        log.info("Starting vLLM: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdout=open(stdout_log, "w"),
            stderr=open(stderr_log, "w"),
            preexec_fn=os.setsid,
        )

    def health_poll(self, timeout_s: float = 900.0, interval_s: float = 5.0) -> bool:
        url = f"http://127.0.0.1:{self.port}/health"
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
            if self._proc is not None and self._proc.poll() is not None:
                return False
            time.sleep(interval_s)
        return False

    def set_env(self) -> None:
        os.environ["TRAILTRAINING_LLM_BASE_URL"] = f"http://127.0.0.1:{self.port}/v1"
        os.environ.setdefault("OPENROUTER_API_KEY", "dummy")

    def shutdown(self, grace_s: float = 15.0) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            self._proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._proc.wait()

    def __enter__(self) -> "VllmServer":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()
