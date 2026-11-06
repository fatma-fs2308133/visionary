"""Classical virtual try-on: Python port of the MATLAB pipeline in ../tryon.

pose (MediaPipe) -> homography warp (from scratch) -> sleeves follow arms
-> body fit (person mask) -> lighting match -> shading transfer
-> feathered blend (from scratch)
"""

from .garment import Garment, load_garment, prepare_garment
from .pipeline import TryOnOptions, try_on

__all__ = ["Garment", "load_garment", "prepare_garment", "TryOnOptions", "try_on"]
