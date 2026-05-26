"""
Qualitative Analysis Suite
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("qualitative-analysis")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
