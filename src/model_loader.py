"""Model loading utilities using trimesh."""

import numpy as np
import trimesh
import trimesh.util


# Supported file extensions and their descriptions
SUPPORTED_FORMATS = {
    ".obj": "Wavefront OBJ",
    ".stl": "Stereolithography STL",
    ".ply": "Polygon File Format PLY",
    ".glb": "Binary glTF",
    ".gltf": "GL Transmission Format",
    ".3mf": "3D Manufacturing Format",
    ".off": "Object File Format OFF",
    ".dae": "Collada DAE",
}

FILE_FILTER = (
    "3D Models ("
    + " ".join(f"*{ext}" for ext in SUPPORTED_FORMATS)
    + ");;"
    + ";;".join(
        f"{desc} (*{ext})" for ext, desc in SUPPORTED_FORMATS.items()
    )
    + ";;All Files (*)"
)


def load_model(filepath: str) -> dict:
    """Load a 3D model from *filepath* and return mesh data suitable for rendering.

    Returns a dict with:
        vertex_array  – flat (N*3, 3) float32 array (one row per triangle vertex)
        normal_array  – flat (N*3, 3) float32 vertex normals
        vertices      – (V, 3) float32 unique vertices
        faces         – (F, 3) int32 face indices
        bbox_min      – (3,) float32 bounding box minimum
        bbox_max      – (3,) float32 bounding box maximum
        vertex_count  – number of unique vertices
        face_count    – number of faces
        filepath      – original file path
    """
    loaded = trimesh.load(filepath, force="mesh")

    if isinstance(loaded, trimesh.Scene):
        meshes = [
            g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)
        ]
        if not meshes:
            raise ValueError("No triangular meshes found in the file.")
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError(f"Unsupported geometry type: {type(loaded)}")

    if len(mesh.faces) == 0:
        raise ValueError("The mesh contains no faces.")

    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)

    # Ensure vertex normals are present
    normals = np.array(mesh.vertex_normals, dtype=np.float32)
    if normals.shape != vertices.shape:
        normals = np.zeros_like(vertices)

    # Build flat arrays: one entry per triangle vertex (for glDrawArrays)
    flat_indices = faces.reshape(-1)
    vertex_array = vertices[flat_indices]
    normal_array = normals[flat_indices]

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)

    return {
        "vertex_array": np.ascontiguousarray(vertex_array),
        "normal_array": np.ascontiguousarray(normal_array),
        "vertices": vertices,
        "faces": faces,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "filepath": filepath,
    }
