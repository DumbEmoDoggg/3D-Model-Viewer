"""Model loading utilities using trimesh and pyassimp."""

import os

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
    ".fbx": "Autodesk FBX",
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


def _compute_tangents(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
) -> np.ndarray:
    """Return per-vertex tangent vectors computed from UV coordinates.

    Parameters
    ----------
    vertices : (V, 3) float32
    faces    : (F, 3) int32
    uvs      : (V, 2) float32

    Returns
    -------
    (V, 3) float32 tangents (already normalised; falls back to (1,0,0) when
    a face has degenerate UVs).
    """
    n = len(vertices)
    tangents = np.zeros((n, 3), dtype=np.float64)

    v0 = vertices[faces[:, 0]].astype(np.float64)
    v1 = vertices[faces[:, 1]].astype(np.float64)
    v2 = vertices[faces[:, 2]].astype(np.float64)

    uv0 = uvs[faces[:, 0]].astype(np.float64)
    uv1 = uvs[faces[:, 1]].astype(np.float64)
    uv2 = uvs[faces[:, 2]].astype(np.float64)

    edge1 = v1 - v0          # (F, 3)
    edge2 = v2 - v0
    d_uv1 = uv1 - uv0        # (F, 2)
    d_uv2 = uv2 - uv0

    det = d_uv1[:, 0] * d_uv2[:, 1] - d_uv2[:, 0] * d_uv1[:, 1]   # (F,)
    inv_det = np.where(np.abs(det) > 1e-10, 1.0 / det, 0.0)

    tang = inv_det[:, None] * (
        d_uv2[:, 1:2] * edge1 - d_uv1[:, 1:2] * edge2
    )   # (F, 3)

    np.add.at(tangents, faces[:, 0], tang)
    np.add.at(tangents, faces[:, 1], tang)
    np.add.at(tangents, faces[:, 2], tang)

    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    default = np.array([[1.0, 0.0, 0.0]])
    tangents = np.where(norms > 1e-10, tangents / np.where(norms > 1e-10, norms, 1.0), default)

    return tangents.astype(np.float32)


def _load_fbx(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, bool]:
    """Load an FBX file using pyassimp and return (vertices, faces, uvs, has_uv).

    All meshes in the scene are merged into one. Returns per-unique-vertex arrays
    ready for the same post-processing path used by the trimesh loader.

    Raises
    ------
    ImportError
        When pyassimp is not installed or the native assimp library is missing.
    ValueError
        When the file contains no triangular geometry.
    """
    try:
        import pyassimp
        import pyassimp.postprocess as pp
    except ImportError as exc:
        raise ImportError(
            "pyassimp is required to load FBX files. "
            "Install it with: pip install pyassimp\n"
            "The native assimp library is also required "
            "(e.g. 'sudo apt install libassimp5' on Debian/Ubuntu)."
        ) from exc

    processing = (
        pp.aiProcess_Triangulate
        | pp.aiProcess_JoinIdenticalVertices
        | pp.aiProcess_GenSmoothNormals
        | pp.aiProcess_FlipUVs
    )

    with pyassimp.load(filepath, processing=processing) as scene:
        if not scene.meshes:
            raise ValueError("No meshes found in the FBX file.")

        all_vertices: list[np.ndarray] = []
        all_faces: list[np.ndarray] = []
        all_uvs: list[np.ndarray | None] = []

        vertex_offset = 0
        any_uv = False

        for mesh in scene.meshes:
            verts = np.array(mesh.vertices, dtype=np.float32)   # (V, 3)
            faces = np.array(mesh.faces, dtype=np.int32)         # (F, 3)

            if len(faces) == 0:
                continue

            # UVs: assimp stores them as (V, 3) with w=0 for 2D coords
            uv: np.ndarray | None = None
            if (
                mesh.texturecoords is not None
                and len(mesh.texturecoords) > 0
                and mesh.texturecoords[0].shape[0] == len(verts)
            ):
                uv = np.array(mesh.texturecoords[0][:, :2], dtype=np.float32)
                any_uv = True

            all_vertices.append(verts)
            all_faces.append(faces + vertex_offset)
            all_uvs.append(uv)
            vertex_offset += len(verts)

        if not all_vertices:
            raise ValueError("No triangular faces found in the FBX file.")

        vertices = np.concatenate(all_vertices, axis=0)
        faces = np.concatenate(all_faces, axis=0)

        uvs: np.ndarray | None = None
        if any_uv:
            # Fill in zero UVs for sub-meshes that had none
            uv_parts: list[np.ndarray] = []
            for i, uv in enumerate(all_uvs):
                if uv is not None:
                    uv_parts.append(uv)
                else:
                    uv_parts.append(
                        np.zeros((len(all_vertices[i]), 2), dtype=np.float32)
                    )
            uvs = np.concatenate(uv_parts, axis=0)

        return vertices, faces, uvs, any_uv


def load_model(filepath: str) -> dict:
    """Load a 3D model from *filepath* and return mesh data suitable for rendering.

    Returns a dict with:
        vertex_array  – flat (N*3, 3) float32 array (one row per triangle vertex)
        normal_array  – flat (N*3, 3) float32 vertex normals
        uv_array      – flat (N*3, 2) float32 UV texture coordinates
        tangent_array – flat (N*3, 3) float32 tangent vectors
        has_uv        – bool, True if the mesh has valid UV coordinates
        vertices      – (V, 3) float32 unique vertices
        faces         – (F, 3) int32 face indices
        bbox_min      – (3,) float32 bounding box minimum
        bbox_max      – (3,) float32 bounding box maximum
        vertex_count  – number of unique vertices
        face_count    – number of faces
        filepath      – original file path
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".fbx":
        vertices, faces, uvs, has_uv = _load_fbx(filepath)
        # Compute vertex normals via trimesh for consistency
        tmp_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        normals = np.array(tmp_mesh.vertex_normals, dtype=np.float32)
        if normals.shape != vertices.shape:
            normals = np.zeros_like(vertices)
    else:
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

        # --- UV coordinates ---
        has_uv = False
        uvs: np.ndarray | None = None
        visual = mesh.visual
        if hasattr(visual, "uv") and visual.uv is not None:
            raw_uv = np.array(visual.uv, dtype=np.float32)
            if raw_uv.shape == (len(vertices), 2):
                uvs = raw_uv
                has_uv = True

    # --- Tangent vectors (needed for normal mapping) ---
    if has_uv and uvs is not None:
        tangents = _compute_tangents(vertices, faces, uvs)
    else:
        tangents = np.tile(
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (len(vertices), 1)
        )

    # Build flat arrays: one entry per triangle vertex (for glDrawArrays)
    flat_indices = faces.reshape(-1)
    vertex_array = vertices[flat_indices]
    normal_array = normals[flat_indices]
    tangent_array = tangents[flat_indices]
    uv_array = (
        uvs[flat_indices]
        if has_uv and uvs is not None
        else np.zeros((len(flat_indices), 2), dtype=np.float32)
    )

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)

    return {
        "vertex_array":  np.ascontiguousarray(vertex_array),
        "normal_array":  np.ascontiguousarray(normal_array),
        "uv_array":      np.ascontiguousarray(uv_array),
        "tangent_array": np.ascontiguousarray(tangent_array),
        "has_uv":        has_uv,
        "vertices":      vertices,
        "faces":         faces,
        "bbox_min":      bbox_min,
        "bbox_max":      bbox_max,
        "vertex_count":  len(vertices),
        "face_count":    len(faces),
        "filepath":      filepath,
    }
