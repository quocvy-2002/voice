"""Thread-safe control state for long-running demo tasks."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Literal


class TaskCancelled(Exception):
    """Raised at a cooperative checkpoint after Stop was requested."""


RunState = Literal[
    "preparing", "running", "cancelling", "cancelled", "completed", "failed"
]
RunOutcome = Literal["cancelled", "completed", "failed"]
ACTIVE_STATES = frozenset({"preparing", "running", "cancelling"})


@dataclass(frozen=True)
class RunSnapshot:
    run_id: str
    state: RunState
    fraction: float | None
    stage: str


class RunControl:
    def __init__(self) -> None:
        self.run_id = uuid.uuid4().hex
        self.cancel_event = threading.Event()
        self._lock = threading.Lock()
        self._state: RunState = "preparing"
        self._fraction: float | None = None
        self._stage = "Đang chuẩn bị…"

    def report(self, fraction: float | None, stage: str) -> None:
        with self._lock:
            if self._state not in ACTIVE_STATES:
                return
            if fraction is not None:
                fraction = min(0.99, max(0.0, float(fraction)))
                self._fraction = max(self._fraction or 0.0, fraction)
            if self._state != "cancelling":
                self._state = "running" if fraction is not None else "preparing"
                self._stage = stage

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise TaskCancelled()

    def request_stop(self) -> bool:
        with self._lock:
            if self._state not in ACTIVE_STATES:
                return False
            self.cancel_event.set()
            self._state = "cancelling"
            self._stage = "Đang dừng…"
            return True

    def finish(self, outcome: RunOutcome) -> None:
        with self._lock:
            if self._state not in ACTIVE_STATES:
                return
            if outcome == "completed" and self.cancel_event.is_set():
                outcome = "cancelled"
            self._state = outcome
            if outcome == "completed":
                self._fraction = 1.0
                self._stage = "Hoàn tất."
            elif outcome == "cancelled":
                self._stage = "Đã dừng."
            else:
                self._stage = "Có lỗi khi xử lý."

    def snapshot(self) -> RunSnapshot:
        with self._lock:
            return RunSnapshot(self.run_id, self._state, self._fraction, self._stage)


class RunRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[tuple[str, str], RunControl] = {}

    def begin(self, session_token: str, workflow: str) -> RunControl | None:
        key = (session_token, workflow)
        with self._lock:
            current = self._runs.get(key)
            if current is not None and current.snapshot().state in ACTIVE_STATES:
                return None
            run = RunControl()
            self._runs[key] = run
            return run

    def get(
        self, session_token: str, workflow: str, run_id: str
    ) -> RunControl | None:
        with self._lock:
            current = self._runs.get((session_token, workflow))
            return current if current is not None and current.run_id == run_id else None

    def request_stop(self, session_token: str, workflow: str, run_id: str) -> bool:
        current = self.get(session_token, workflow, run_id)
        return current.request_stop() if current is not None else False

    def finish(
        self,
        session_token: str,
        workflow: str,
        run_id: str,
        outcome: RunOutcome,
    ) -> bool:
        with self._lock:
            current = self._runs.get((session_token, workflow))
            if current is None or current.run_id != run_id:
                return False
            current.finish(outcome)
            return True

    def snapshot(self, session_token: str, workflow: str) -> RunSnapshot | None:
        with self._lock:
            current = self._runs.get((session_token, workflow))
            return current.snapshot() if current is not None else None

    def drop_session(self, session_token: str) -> None:
        with self._lock:
            keys = [key for key in self._runs if key[0] == session_token]
            runs = [self._runs.pop(key) for key in keys]
        for run in runs:
            run.request_stop()
