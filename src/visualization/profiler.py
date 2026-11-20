"""High-resolution frame-by-frame latency profiler for real-time computer vision pipelines."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class StageMetrics:
    """Rolling latency metrics for an individual pipeline stage."""

    history: deque[float] = field(default_factory=lambda: deque(maxlen=60))

    def record(self, duration_ms: float) -> None:
        self.history.append(duration_ms)

    @property
    def average_ms(self) -> float:
        return sum(self.history) / len(self.history) if self.history else 0.0


class PipelineProfiler:
    """Measures end-to-end latency, stage breakdown, and smoothed FPS without allocation overhead."""

    def __init__(self, window_size: int = 60) -> None:
        self.window_size = window_size
        self.stages: Dict[str, StageMetrics] = {
            "camera_read": StageMetrics(deque(maxlen=window_size)),
            "hand_detect": StageMetrics(deque(maxlen=window_size)),
            "feature_extract": StageMetrics(deque(maxlen=window_size)),
            "ml_inference": StageMetrics(deque(maxlen=window_size)),
            "rendering": StageMetrics(deque(maxlen=window_size)),
        }
        self._frame_times: deque[float] = deque(maxlen=window_size)
        self._prev_frame_timestamp = time.perf_counter()
        self._active_stage_start: Optional[float] = None
        self._active_stage_name: Optional[str] = None

    def tick_frame(self) -> float:
        """Call at the start of each frame. Returns instantaneous FPS."""
        now = time.perf_counter()
        dt = now - self._prev_frame_timestamp
        self._prev_frame_timestamp = now
        if dt > 0:
            self._frame_times.append(1.0 / dt)
        return self.fps

    @property
    def fps(self) -> float:
        """Moving average frames per second."""
        return sum(self._frame_times) / len(self._frame_times) if self._frame_times else 0.0

    @property
    def total_latency_ms(self) -> float:
        """Sum of average latencies across all registered stages."""
        return sum(s.average_ms for s in self.stages.values())

    def start_stage(self, name: str) -> None:
        """Begin measuring a specific pipeline stage."""
        self._active_stage_name = name
        self._active_stage_start = time.perf_counter()

    def end_stage(self) -> float:
        """Conclude measuring the active pipeline stage. Returns elapsed milliseconds."""
        if self._active_stage_start is None or self._active_stage_name is None:
            return 0.0

        elapsed_ms = (time.perf_counter() - self._active_stage_start) * 1000.0
        if self._active_stage_name in self.stages:
            self.stages[self._active_stage_name].record(elapsed_ms)

        self._active_stage_start = None
        self._active_stage_name = None
        return elapsed_ms

    def get_summary(self) -> Dict[str, float]:
        """Return a mapping of stage names to average duration in milliseconds."""
        summary = {name: stage.average_ms for name, stage in self.stages.items()}
        summary["total_ms"] = self.total_latency_ms
        summary["fps"] = self.fps
        return summary
