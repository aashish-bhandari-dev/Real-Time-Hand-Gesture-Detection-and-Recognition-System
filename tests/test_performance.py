"""Unit tests for pipeline profiler and rendering performance optimizations."""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.detection.hand_detector import DetectedHand, LandmarkPoint
from src.visualization.drawer import Visualizer
from src.visualization.profiler import PipelineProfiler


def test_profiler_tick_and_fps():
    """Verify profiler records frame intervals and computes moving average FPS."""
    profiler = PipelineProfiler(window_size=10)

    for _ in range(5):
        time.sleep(0.005)
        profiler.tick_frame()

    assert profiler.fps > 0.0
    assert len(profiler._frame_times) == 5


def test_profiler_stage_measurement():
    """Verify measuring individual pipeline stages."""
    profiler = PipelineProfiler(window_size=10)

    profiler.start_stage("ml_inference")
    time.sleep(0.005)
    elapsed = profiler.end_stage()

    assert elapsed >= 4.0  # At least 4ms
    assert profiler.stages["ml_inference"].average_ms >= 4.0

    summary = profiler.get_summary()
    assert "ml_inference" in summary
    assert "total_ms" in summary
    assert "fps" in summary


def test_optimized_drawer_overlays():
    """Verify sliced alpha blending operates on image frames without shape mutations."""
    visualizer = Visualizer()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Diagnostic panel
    stage_latencies = {"camera_read": 3.2, "hand_detect": 12.5, "ml_inference": 0.8}
    frame_annotated = visualizer.draw_diagnostics_panel(frame, stage_latencies)
    assert frame_annotated.shape == (720, 1280, 3)

    # Bounding box with confidence bar and stability indicator
    landmarks = [LandmarkPoint(x=0.4 + (i % 5) * 0.04, y=0.4 + (i // 5) * 0.04, z=0.0) for i in range(21)]
    hand = DetectedHand(handedness="Right", confidence=0.95, landmarks=landmarks)
    frame_boxed = visualizer.draw_bounding_box(
        frame=frame,
        hand=hand,
        label="Peace Sign",
        confidence=0.92,
        is_stable=True,
    )
    assert frame_boxed.shape == (720, 1280, 3)
