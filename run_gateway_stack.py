"""Start the Legacy API, Model API, and Core Gateway on ports 8000-8002."""

import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SERVICES = [
    ("Legacy API :8000", "backend.main"),
    ("Model API :8001", "backend.model_api"),
    ("Core Gateway :8002", "backend.core_gateway"),
]


def build_child_environment(parent_environment: dict[str, str] | None = None) -> dict[str, str]:
    """Copy the caller's environment so explicit debug gates reach each service."""
    env = dict(os.environ if parent_environment is None else parent_environment)
    env.setdefault("LEGACY_CORE_API_URL", "http://127.0.0.1:8000/api/v1/chat/completions")
    env.setdefault("CORE_API_HOST", "127.0.0.1")
    env.setdefault("MODEL_API_HOST", "127.0.0.1")
    env.setdefault("CORE_GATEWAY_HOST", "127.0.0.1")
    env.setdefault("CORE_GATEWAY_PORT", "8002")
    return env


def main() -> int:
    env = build_child_environment()

    processes: list[tuple[str, subprocess.Popen]] = []
    try:
        for service_name, module_name in SERVICES:
            print(f"Starting {service_name}...", flush=True)
            process = subprocess.Popen(
                [sys.executable, "-m", module_name],
                cwd=PROJECT_ROOT,
                env=env,
            )
            processes.append((service_name, process))
            time.sleep(1)

        print(
            "Services started: Legacy API :8000, Model API :8001, "
            "Core Gateway :8002. Press Ctrl+C to stop all.",
            flush=True,
        )

        while True:
            for service_name, process in processes:
                exit_code = process.poll()
                if exit_code is not None:
                    print(
                        f"{service_name} exited with code {exit_code}; stopping stack.",
                        flush=True,
                    )
                    return exit_code or 1
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping services...", flush=True)
        return 0
    finally:
        for _, process in reversed(processes):
            if process.poll() is None:
                process.terminate()
        for _, process in reversed(processes):
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
