"""CauchyLift: Curvature-Adaptive Matrix Optimizer with Historical Momentum and Decoupled Weight Decay."""

from .optimizer import CauchyLift
from .reference import cauchylift_direction, cauchylift_reference_step
from .xla import (
    cauchylift_direction_xla,
    cauchylift_xla_foreach_step_,
    cauchylift_xla_step_,
    get_hlo_text,
    get_tpu_device,
    is_tpu_available,
    sync_tpu,
)

__all__ = [
    "CauchyLift",
    "cauchylift_direction",
    "cauchylift_reference_step",
    "cauchylift_direction_xla",
    "cauchylift_xla_step_",
    "cauchylift_xla_foreach_step_",
    "get_hlo_text",
    "get_tpu_device",
    "is_tpu_available",
    "sync_tpu",
]
__version__ = "1.0.0"
