"""
Figurative language analysis module.
"""

from .detector import FigurativeDetector
from .models import DetectionResult, Instance

__all__ = ["FigurativeDetector", "DetectionResult", "Instance"]
