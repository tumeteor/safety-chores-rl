"""Persistent spawned workers with bounded calls and owned-process cleanup."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import os
import signal
import subprocess
import threading
import time

from .runtime import build_runtime, execute
from .types import RolloutResult


def _worker(connection, spec):
    if os.name == "posix":
        os.setsid()  # Native children share an owned process group for timeout cleanup.
    runtime = None
    try:
        connection.send(("started", None))
        runtime = build_runtime(spec)
        connection.send(("ready", None))
        while True:
            request = connection.recv()
            if request is None:
                break
            connection.send(("result", execute(runtime, request)))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send(("error", f"{type(exc).__name__}: {exc}"))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        if runtime is not None:
            runtime.close()
        connection.close()


class WorkerSlot:
    def __init__(self, spec):
        self.spec = spec
        self.process = self.connection = None
        self.group_ready = False

    def _stop(self):
        process = self.process
        if process is None:
            return
        if os.name == "posix" and self.group_ready:
            try:
                # The group can still own Unity children after its leader exits.
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.is_alive():
            if os.name == "posix":
                try:
                    if os.getpgid(process.pid) == process.pid:
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except ProcessLookupError:
                    pass
            elif os.name == "nt":
                # PID comes only from our multiprocessing.Process, never user text.
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if process.is_alive():
                    process.kill()
            else:
                process.kill()
        process.join(timeout=3)
        self.connection.close()
        if not process.is_alive():
            process.close()
        self.process = self.connection = None
        self.group_ready = False

    def _receive(self, deadline, cancel):
        while time.monotonic() < deadline and not cancel.is_set():
            if self.connection.poll(min(0.02, max(0, deadline - time.monotonic()))):
                return self.connection.recv()
            if not self.process.is_alive():
                raise RuntimeError("Rollout worker exited without a result")
        raise TimeoutError("cancelled" if cancel.is_set() else "timeout")

    def call(self, request, deadline, cancel):
        started = time.monotonic()
        deadline = min(deadline, started + request.budget.timeout_seconds)
        try:
            if self.process is None:
                context = mp.get_context("spawn")
                self.connection, child = context.Pipe()
                self.process = context.Process(target=_worker, args=(child, self.spec), daemon=False)
                self.process.start()
                child.close()
                kind, value = self._receive(deadline, cancel)
                if kind != "started":
                    raise RuntimeError(value)
                self.group_ready = os.name == "posix"
                kind, value = self._receive(deadline, cancel)
                if kind != "ready":
                    raise RuntimeError(value)
            if cancel.is_set() or time.monotonic() >= deadline:
                raise TimeoutError("cancelled" if cancel.is_set() else "timeout")
            self.connection.send(request)
            kind, value = self._receive(deadline, cancel)
            if kind != "result":
                raise RuntimeError(value)
            return value
        except TimeoutError as exc:
            self._stop()
            return RolloutResult.failed(request, str(exc), str(exc), native_steps=request.reservation)
        except Exception as exc:
            self._stop()
            result = RolloutResult.failed(request, "error", f"worker_error: {exc}",
                                          native_steps=request.reservation)
            return result

    def close(self):
        if self.process is not None:
            try:
                self.connection.send(None)
                self.process.join(timeout=1)
            except (EOFError, BrokenPipeError, OSError):
                pass
            self._stop()


class WorkerPool:
    """A wave has at most one request per slot; result order follows input order."""
    def __init__(self, spec, workers):
        self.slots = [WorkerSlot(spec) for _ in range(workers)]
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rollout-control")

    def evaluate(self, requests, deadline, cancel):
        futures = [self.executor.submit(slot.call, request, deadline, cancel)
                   for slot, request in zip(self.slots, requests)]
        return [future.result() for future in futures]

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        for slot in self.slots:
            slot.close()
