"""Material data class for mesh rendering."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Material:
    """PBR-lite material properties for a 3-D mesh.

    Textures are specified as file-system paths to common image files
    (PNG, JPG, BMP, …).  When a texture path is *None* the corresponding
    scalar value is used instead.
    """

    # Base colour – RGBA in [0, 1].  Used when *base_color_texture* is None.
    base_color: Tuple[float, float, float, float] = (0.78, 0.70, 0.60, 1.0)
    # Path to a base-colour / albedo texture image.
    base_color_texture: Optional[str] = None

    # Path to a tangent-space normal-map texture image.
    # Requires the mesh to have UV texture coordinates.
    normal_map_texture: Optional[str] = None

    # Smoothness value in [0, 1] (0 = fully rough, 1 = mirror-smooth).
    # Used when *smoothness_texture* is None.
    smoothness: float = 0.5
    # Path to a smoothness texture image (R channel, greyscale).
    smoothness_texture: Optional[str] = None

    # Metallic value in [0, 1] (0 = dielectric, 1 = fully metallic).
    # Used when *metallic_texture* is None.
    metallic: float = 0.0
    # Path to a metallic texture image (R channel, greyscale).
    metallic_texture: Optional[str] = None
