"""Typed discovery for the supported real-G1 tasks; the YAM registry with G1 entries."""

from __future__ import annotations

from cap_harness.yam_real.registry import YamRealTaskMetadata, YamRealTaskRegistry

G1_REAL_TASKS: tuple[YamRealTaskMetadata, ...] = (
    YamRealTaskMetadata(
        "observe_station", "Observe the G1 station.", family="g1_real", scene_reset="none"
    ),
    YamRealTaskMetadata(
        "reach_home",
        "Return both arms to the station home pose.",
        family="g1_real",
        scene_reset="none",
    ),
    YamRealTaskMetadata(
        "joint_motion", "Execute a bounded joint motion.", family="g1_real", scene_reset="none"
    ),
)


class G1RealTaskRegistry(YamRealTaskRegistry):
    """Resolve only the explicitly supported real-G1 task set."""

    def __init__(self, tasks: tuple[YamRealTaskMetadata, ...] = G1_REAL_TASKS) -> None:
        super().__init__(tasks)


__all__ = ["G1_REAL_TASKS", "G1RealTaskRegistry"]
