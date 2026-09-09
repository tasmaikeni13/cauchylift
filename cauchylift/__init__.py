"""CauchyLift: Curvature-Adaptive Matrix Optimizer with Historical Momentum and Decoupled Weight Decay."""

from .optimizer import CauchyLift
from .reference import cauchylift_direction, cauchylift_reference_step

__all__ = ["CauchyLift", "cauchylift_direction", "cauchylift_reference_step"]
__version__ = "1.0.0"
