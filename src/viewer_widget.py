"""OpenGL 3-D viewer widget with orbit / pan / zoom controls."""

import math
from typing import Optional

import numpy as np
from OpenGL.GL import (
    # --- legacy fixed-function (used for grid / axes / fallback mesh) ---
    GL_AMBIENT,
    GL_AMBIENT_AND_DIFFUSE,
    GL_COLOR_BUFFER_BIT,
    GL_COLOR_MATERIAL,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_DIFFUSE,
    GL_FLOAT,
    GL_FRONT_AND_BACK,
    GL_LIGHT0,
    GL_LIGHTING,
    GL_LINE,
    GL_LINES,
    GL_NORMALIZE,
    GL_POSITION,
    GL_SHININESS,
    GL_SPECULAR,
    GL_TRIANGLES,
    GL_VERTEX_ARRAY,
    GL_NORMAL_ARRAY,
    GL_FILL,
    GL_POLYGON_OFFSET_FILL,
    glBegin,
    glClear,
    glClearColor,
    glColor3f,
    glColorMaterial,
    glDisable,
    glDisableClientState,
    glDrawArrays,
    glEnable,
    glEnableClientState,
    glEnd,
    glLightfv,
    glLineWidth,
    glLoadIdentity,
    glLoadMatrixf,
    glMaterialf,
    glMaterialfv,
    glNormalPointer,
    glPolygonMode,
    glPolygonOffset,
    glVertex3f,
    glVertexPointer,
    glViewport,
    glMatrixMode,
    GL_PROJECTION,
    GL_MODELVIEW,
    # --- shader pipeline ---
    GL_VERTEX_SHADER,
    GL_FRAGMENT_SHADER,
    GL_COMPILE_STATUS,
    GL_LINK_STATUS,
    GL_FALSE,
    GL_TRUE,
    glCreateShader,
    glCreateProgram,
    glShaderSource,
    glCompileShader,
    glGetShaderiv,
    glGetShaderInfoLog,
    glAttachShader,
    glBindAttribLocation,
    glLinkProgram,
    glGetProgramiv,
    glGetProgramInfoLog,
    glUseProgram,
    glDeleteShader,
    glGetUniformLocation,
    glUniform1i,
    glUniform1f,
    glUniform4f,
    glUniformMatrix3fv,
    glUniformMatrix4fv,
    glEnableVertexAttribArray,
    glDisableVertexAttribArray,
    glVertexAttribPointer,
    # --- textures ---
    GL_TEXTURE_2D,
    GL_TEXTURE0,
    GL_TEXTURE1,
    GL_TEXTURE2,
    GL_TEXTURE3,
    GL_RGBA,
    GL_UNSIGNED_BYTE,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_LINEAR,
    GL_LINEAR_MIPMAP_LINEAR,
    GL_REPEAT,
    GL_UNPACK_ALIGNMENT,
    glGenTextures,
    glBindTexture,
    glTexImage2D,
    glTexParameteri,
    glDeleteTextures,
    glActiveTexture,
    glPixelStorei,
    glGenerateMipmap,
)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QOpenGLWidget

from material import Material


# ---------------------------------------------------------------------------
# Pure-numpy replacements for deprecated GLU functions
# ---------------------------------------------------------------------------

def _perspective_matrix(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Build a column-major OpenGL perspective projection matrix."""
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[3, 2] = (2.0 * far * near) / (near - far)
    m[2, 3] = -1.0
    return m


def _look_at_matrix(eye, center, up) -> np.ndarray:
    """Build a column-major OpenGL look-at (view) matrix."""
    eye    = np.array(eye,    dtype=np.float64)
    center = np.array(center, dtype=np.float64)
    up     = np.array(up,     dtype=np.float64)

    f = center - eye
    fn = np.linalg.norm(f)
    if fn < 1e-10:
        return np.eye(4, dtype=np.float32)
    f /= fn

    s = np.cross(f, up)
    sn = np.linalg.norm(s)
    if sn < 1e-10:
        # The requested up vector is parallel to the viewing direction (e.g., looking
        # straight up or down along Y).  Fall back to the world Z axis so the camera
        # still produces a valid orthonormal frame.  The camera in this application
        # always orbits with elevation clamped to +/-89 deg so this path is only reached
        # when the caller explicitly passes a degenerate up vector.
        up = np.array([0.0, 0.0, 1.0])
        s = np.cross(f, up)
        sn = np.linalg.norm(s)
    s /= sn
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, 0] = s[0]; m[0, 1] = s[1]; m[0, 2] = s[2]
    m[1, 0] = u[0]; m[1, 1] = u[1]; m[1, 2] = u[2]
    m[2, 0] =-f[0]; m[2, 1] =-f[1]; m[2, 2] =-f[2]
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] =  np.dot(f, eye)
    # OpenGL expects column-major, so transpose before passing
    return m.T.astype(np.float32)


# ---------------------------------------------------------------------------
# GLSL shader sources
# ---------------------------------------------------------------------------

_VERT_SRC = """
#version 120

attribute vec3 a_position;
attribute vec3 a_normal;
attribute vec2 a_texcoord;
attribute vec3 a_tangent;

varying vec3 v_pos_eye;
varying vec3 v_normal_eye;
varying vec2 v_texcoord;
varying vec3 v_tangent_eye;
varying vec3 v_bitangent_eye;

uniform mat4 u_mv;
uniform mat4 u_mvp;
uniform mat3 u_normal_mat;

void main() {
    vec4 pos_eye    = u_mv * vec4(a_position, 1.0);
    v_pos_eye       = pos_eye.xyz;

    vec3 N          = normalize(u_normal_mat * a_normal);
    v_normal_eye    = N;
    v_texcoord      = a_texcoord;

    // Build tangent and bitangent in eye space for normal mapping
    vec3 T          = normalize(u_normal_mat * a_tangent);
    T               = normalize(T - dot(T, N) * N);   // re-orthogonalise
    v_tangent_eye   = T;
    v_bitangent_eye = cross(N, T);

    gl_Position = u_mvp * vec4(a_position, 1.0);
}
"""

_FRAG_SRC = """
#version 120

varying vec3 v_pos_eye;
varying vec3 v_normal_eye;
varying vec2 v_texcoord;
varying vec3 v_tangent_eye;
varying vec3 v_bitangent_eye;

uniform sampler2D u_tex_base_color;
uniform sampler2D u_tex_normal;
uniform sampler2D u_tex_smoothness;
uniform sampler2D u_tex_metallic;

uniform vec4  u_base_color;
uniform float u_smoothness_val;
uniform float u_metallic_val;

// 1 = use this texture, 0 = use scalar value
uniform int u_use_tex_base_color;
uniform int u_use_tex_normal;
uniform int u_use_tex_smoothness;
uniform int u_use_tex_metallic;

// 1 = the mesh has valid UV texture coordinates
uniform int u_has_uv;

void main() {
    // ---- base colour ----
    vec4 albedo;
    if (u_use_tex_base_color == 1 && u_has_uv == 1) {
        albedo = texture2D(u_tex_base_color, v_texcoord);
    } else {
        albedo = u_base_color;
    }

    // ---- normal ----
    vec3 N;
    if (u_use_tex_normal == 1 && u_has_uv == 1) {
        vec3 nm  = texture2D(u_tex_normal, v_texcoord).rgb * 2.0 - 1.0;
        mat3 TBN = mat3(
            normalize(v_tangent_eye),
            normalize(v_bitangent_eye),
            normalize(v_normal_eye)
        );
        N = normalize(TBN * nm);
    } else {
        N = normalize(v_normal_eye);
    }

    // ---- smoothness ----
    float sm;
    if (u_use_tex_smoothness == 1 && u_has_uv == 1) {
        sm = texture2D(u_tex_smoothness, v_texcoord).r;
    } else {
        sm = u_smoothness_val;
    }

    // ---- metallic ----
    float metal;
    if (u_use_tex_metallic == 1 && u_has_uv == 1) {
        metal = texture2D(u_tex_metallic, v_texcoord).r;
    } else {
        metal = u_metallic_val;
    }

    // ---- Blinn-Phong lighting (headlight at camera = eye-space origin) ----
    // Camera is at (0,0,0) in eye space; the light is co-located with the camera.
    vec3 L     = normalize(-v_pos_eye);   // direction to light
    vec3 V     = normalize(-v_pos_eye);   // direction to viewer (same – headlight)
    vec3 H     = normalize(L + V);        // half-way vector (= L since L == V)

    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);

    // Shininess derived from smoothness so that smooth=1 => sharp specular
    float roughness = max(0.01, 1.0 - sm);
    float shininess = max(1.0, 2.0 / (roughness * roughness) - 2.0);

    // PBR-lite: F0 = lerp(0.04, albedo, metallic)
    vec3 F0      = mix(vec3(0.04), albedo.rgb, metal);

    vec3 ambient  = 0.25 * albedo.rgb;
    vec3 diffuse  = 0.85 * NdotL * (1.0 - metal) * albedo.rgb;
    vec3 specular = 0.50 * pow(NdotH, shininess) * F0;

    gl_FragColor = vec4(ambient + diffuse + specular, albedo.a);
}
"""


# ---------------------------------------------------------------------------
# Viewer widget
# ---------------------------------------------------------------------------

class ViewerWidget(QOpenGLWidget):
    """A QOpenGLWidget that renders a triangular mesh with orbit/pan/zoom controls.

    Left-drag   : orbit
    Right-drag  : pan
    Scroll      : zoom
    """

    # Near/far plane are derived from the camera radius to keep the depth
    # buffer accurate across a wide range of model scales.
    NEAR_PLANE_FACTOR = 0.001   # near = radius * NEAR_PLANE_FACTOR
    FAR_PLANE_FACTOR  = 100.0   # far  = radius * FAR_PLANE_FACTOR

    def __init__(self, parent=None):
        super().__init__(parent)

        self.mesh_data = None

        # Camera state (spherical coordinates around *target*)
        self.azimuth = 45.0    # degrees, horizontal rotation
        self.elevation = 30.0  # degrees, vertical rotation (-90..90)
        self.radius = 5.0      # distance from target
        self.target = [0.0, 0.0, 0.0]

        # Mouse interaction state
        self._last_mouse = QPoint()
        self._mouse_button = None

        # Display toggles
        self.show_wireframe = False
        self.show_overlay_wireframe = False
        self.show_axes = True
        self.show_grid = True

        # Material / shader state
        self._material = None          # type: Optional[Material]
        self._shader_program = None    # type: Optional[int]
        self._shader_uniforms = {}     # type: dict
        self._textures = {}            # role -> GL texture id

        self.setMinimumSize(400, 300)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAcceptDrops(True)

    # ------------------------------------------------------------------
    # Qt / OpenGL lifecycle
    # ------------------------------------------------------------------

    def initializeGL(self):
        glClearColor(0.18, 0.18, 0.22, 1.0)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_NORMALIZE)

        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.25, 0.25, 0.25, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.85, 0.85, 0.85, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.50, 0.50, 0.50, 1.0])

        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.4, 0.4, 0.4, 1.0])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 40.0)

        self._compile_shaders()

    def resizeGL(self, w, h):
        if h == 0:
            h = 1
        self._viewport_w = w
        self._viewport_h = h
        glViewport(0, 0, w, h)
        self._update_projection()

    def _update_projection(self):
        w = getattr(self, "_viewport_w", self.width())
        h = getattr(self, "_viewport_h", self.height()) or 1
        near = max(0.001, self.radius * self.NEAR_PLANE_FACTOR)
        far  = self.radius * self.FAR_PLANE_FACTOR
        proj = _perspective_matrix(45.0, w / h, near, far)
        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(proj)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Refresh projection in case radius has changed since last resize
        self._update_projection()

        eye = self._camera_position()
        view = _look_at_matrix(
            eye,
            self.target,
            [0.0, 1.0, 0.0],
        )
        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(view)

        # Keep light attached to the camera position so it always illuminates
        glLightfv(GL_LIGHT0, GL_POSITION, [eye[0], eye[1], eye[2], 1.0])

        if self.show_grid:
            self._draw_grid()

        if self.show_axes:
            self._draw_axes()

        if self.mesh_data is not None:
            self._draw_mesh(view)

    # ------------------------------------------------------------------
    # Shader compilation
    # ------------------------------------------------------------------

    def _compile_shaders(self):
        """Compile and link the GLSL shader program.  Falls back silently."""
        try:
            vert = glCreateShader(GL_VERTEX_SHADER)
            glShaderSource(vert, _VERT_SRC)
            glCompileShader(vert)
            if not glGetShaderiv(vert, GL_COMPILE_STATUS):
                raise RuntimeError(
                    "Vertex shader compile error:\n"
                    + glGetShaderInfoLog(vert).decode(errors="replace")
                )

            frag = glCreateShader(GL_FRAGMENT_SHADER)
            glShaderSource(frag, _FRAG_SRC)
            glCompileShader(frag)
            if not glGetShaderiv(frag, GL_COMPILE_STATUS):
                raise RuntimeError(
                    "Fragment shader compile error:\n"
                    + glGetShaderInfoLog(frag).decode(errors="replace")
                )

            prog = glCreateProgram()
            glAttachShader(prog, vert)
            glAttachShader(prog, frag)

            # Bind attribute locations before linking
            glBindAttribLocation(prog, 0, "a_position")
            glBindAttribLocation(prog, 1, "a_normal")
            glBindAttribLocation(prog, 2, "a_texcoord")
            glBindAttribLocation(prog, 3, "a_tangent")

            glLinkProgram(prog)
            if not glGetProgramiv(prog, GL_LINK_STATUS):
                raise RuntimeError(
                    "Shader link error:\n"
                    + glGetProgramInfoLog(prog).decode(errors="replace")
                )

            glDeleteShader(vert)
            glDeleteShader(frag)

            self._shader_program = int(prog)
            # Cache uniform locations
            for name in (
                "u_mv", "u_mvp", "u_normal_mat",
                "u_tex_base_color", "u_tex_normal",
                "u_tex_smoothness", "u_tex_metallic",
                "u_base_color",
                "u_smoothness_val", "u_metallic_val",
                "u_use_tex_base_color", "u_use_tex_normal",
                "u_use_tex_smoothness", "u_use_tex_metallic",
                "u_has_uv",
            ):
                self._shader_uniforms[name] = glGetUniformLocation(prog, name)

        except Exception as exc:
            import traceback
            traceback.print_exc()
            print("[ViewerWidget] Shader init failed, using fixed-function: %s" % exc)
            self._shader_program = None

    # ------------------------------------------------------------------
    # Texture management
    # ------------------------------------------------------------------

    def _upload_texture(self, path):
        """Load *path* as an OpenGL texture and return its texture id."""
        img = QImage(path)
        if img.isNull():
            raise ValueError("Cannot load image: %s" % path)
        # Flip vertically – OpenGL's origin is bottom-left
        img = img.convertToFormat(QImage.Format_RGBA8888).mirrored(False, True)
        w, h = img.width(), img.height()
        ptr = img.bits()
        ptr.setsize(w * h * 4)
        data = bytes(ptr)

        tex_id = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
            GL_RGBA, GL_UNSIGNED_BYTE, data,
        )
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)
        return tex_id

    def _clear_material_textures(self):
        """Delete all uploaded material textures from GPU memory."""
        for tex_id in self._textures.values():
            try:
                glDeleteTextures(1, [int(tex_id)])
            except Exception:
                pass
        self._textures.clear()

    def _upload_material_textures(self, mat):
        """Upload all textures referenced in *mat* to the GPU."""
        slots = {
            "base_color":  mat.base_color_texture,
            "normal":      mat.normal_map_texture,
            "smoothness":  mat.smoothness_texture,
            "metallic":    mat.metallic_texture,
        }
        for role, path in slots.items():
            if path:
                try:
                    self._textures[role] = self._upload_texture(path)
                except Exception as exc:
                    print("[ViewerWidget] Cannot load texture '%s': %s" % (path, exc))

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _camera_position(self):
        phi   = math.radians(self.azimuth)
        theta = math.radians(self.elevation)
        x = self.target[0] + self.radius * math.cos(theta) * math.sin(phi)
        y = self.target[1] + self.radius * math.sin(theta)
        z = self.target[2] + self.radius * math.cos(theta) * math.cos(phi)
        return [x, y, z]

    def _draw_grid(self):
        glDisable(GL_LIGHTING)
        grid_count = 10
        step = max(0.1, self.radius / 5.0)
        half = grid_count * step
        glColor3f(0.35, 0.35, 0.40)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        for i in range(-grid_count, grid_count + 1):
            t = i * step
            # Lines along Z
            glVertex3f(t,    0, -half)
            glVertex3f(t,    0,  half)
            # Lines along X
            glVertex3f(-half, 0, t)
            glVertex3f( half, 0, t)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_axes(self):
        glDisable(GL_LIGHTING)
        axis_len = max(0.5, self.radius * 0.2)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor3f(0.9, 0.2, 0.2); glVertex3f(0, 0, 0); glVertex3f(axis_len, 0, 0)
        glColor3f(0.2, 0.9, 0.2); glVertex3f(0, 0, 0); glVertex3f(0, axis_len, 0)
        glColor3f(0.2, 0.2, 0.9); glVertex3f(0, 0, 0); glVertex3f(0, 0, axis_len)
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    def _draw_mesh(self, view):
        """Dispatch to shader or fixed-function mesh rendering."""
        if self._material is not None and self._shader_program is not None:
            self._draw_mesh_material(view)
        else:
            self._draw_mesh_basic()

    # ------------------------------------------------------------------
    # Fixed-function mesh drawing (no material / fallback)
    # ------------------------------------------------------------------

    def _draw_mesh_basic(self):
        va = self.mesh_data["vertex_array"]
        na = self.mesh_data["normal_array"]
        n  = len(va)

        if self.show_wireframe:
            # Pure wireframe
            glDisable(GL_LIGHTING)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glColor3f(0.85, 0.85, 0.85)
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, va)
            glDrawArrays(GL_TRIANGLES, 0, n)
            glDisableClientState(GL_VERTEX_ARRAY)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glEnable(GL_LIGHTING)
        else:
            # Solid fill
            glColor3f(0.78, 0.70, 0.60)
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_NORMAL_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, va)
            glNormalPointer(GL_FLOAT, 0, na)
            glDrawArrays(GL_TRIANGLES, 0, n)
            glDisableClientState(GL_NORMAL_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)

            if self.show_overlay_wireframe:
                # Draw wireframe on top
                glDisable(GL_LIGHTING)
                glEnable(GL_POLYGON_OFFSET_FILL)
                glPolygonOffset(1.0, 1.0)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glColor3f(0.15, 0.15, 0.15)
                glLineWidth(0.8)
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(3, GL_FLOAT, 0, va)
                glDrawArrays(GL_TRIANGLES, 0, n)
                glDisableClientState(GL_VERTEX_ARRAY)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                glDisable(GL_POLYGON_OFFSET_FILL)
                glLineWidth(1.0)
                glEnable(GL_LIGHTING)

    # ------------------------------------------------------------------
    # Shader-based mesh drawing (with material)
    # ------------------------------------------------------------------

    def _draw_mesh_material(self, view):
        """Render the mesh using GLSL shaders with the active material."""
        md   = self.mesh_data
        va   = md["vertex_array"]    # (N, 3) float32
        na   = md["normal_array"]    # (N, 3) float32
        uva  = md["uv_array"]        # (N, 2) float32
        ta   = md["tangent_array"]   # (N, 3) float32
        n    = len(va)
        has_uv = int(md.get("has_uv", False))

        mat = self._material

        # Compute MVP: view @ proj gives the right GL column-major MVP
        w = getattr(self, "_viewport_w", self.width())
        h = getattr(self, "_viewport_h", self.height()) or 1
        near = max(0.001, self.radius * self.NEAR_PLANE_FACTOR)
        far  = self.radius * self.FAR_PLANE_FACTOR
        proj = _perspective_matrix(45.0, w / h, near, far)
        mvp  = np.ascontiguousarray(view @ proj, dtype=np.float32)

        # Normal matrix: pass view[:3,:3] so GLSL receives the correct
        # upper-left 3x3 of the math view matrix (V_math[:3,:3]).
        normal_mat = np.ascontiguousarray(view[:3, :3].copy(), dtype=np.float32)

        glUseProgram(self._shader_program)
        u = self._shader_uniforms

        glUniformMatrix4fv(u["u_mv"],         1, GL_FALSE, view)
        glUniformMatrix4fv(u["u_mvp"],        1, GL_FALSE, mvp)
        glUniformMatrix3fv(u["u_normal_mat"], 1, GL_FALSE, normal_mat)

        # ---- material scalar uniforms ----
        r, g, b, a = mat.base_color
        glUniform4f(u["u_base_color"],     r, g, b, a)
        glUniform1f(u["u_smoothness_val"], mat.smoothness)
        glUniform1f(u["u_metallic_val"],   mat.metallic)
        glUniform1i(u["u_has_uv"],         has_uv)

        # ---- bind textures to units 0-3 ----
        tex_slots = [
            ("base_color", "u_tex_base_color", "u_use_tex_base_color", GL_TEXTURE0, 0),
            ("normal",     "u_tex_normal",     "u_use_tex_normal",     GL_TEXTURE1, 1),
            ("smoothness", "u_tex_smoothness", "u_use_tex_smoothness", GL_TEXTURE2, 2),
            ("metallic",   "u_tex_metallic",   "u_use_tex_metallic",   GL_TEXTURE3, 3),
        ]
        for role, tex_uni, flag_uni, gl_unit, unit_idx in tex_slots:
            glUniform1i(u[tex_uni], unit_idx)
            if role in self._textures:
                glActiveTexture(gl_unit)
                glBindTexture(GL_TEXTURE_2D, self._textures[role])
                glUniform1i(u[flag_uni], 1)
            else:
                glUniform1i(u[flag_uni], 0)

        # ---- vertex attribute arrays ----
        glDisable(GL_LIGHTING)
        if self.show_wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, va)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, na)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, uva)
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, ta)

        glDrawArrays(GL_TRIANGLES, 0, n)

        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)
        glDisableVertexAttribArray(2)
        glDisableVertexAttribArray(3)

        if self.show_wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # Unbind textures
        for _role, _tex_uni, _flag_uni, gl_unit, _unit_idx in tex_slots:
            glActiveTexture(gl_unit)
            glBindTexture(GL_TEXTURE_2D, 0)

        glActiveTexture(GL_TEXTURE0)
        glUseProgram(0)
        glEnable(GL_LIGHTING)

        # Wireframe overlay on top of shaded mesh
        if self.show_overlay_wireframe:
            glDisable(GL_LIGHTING)
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(1.0, 1.0)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glColor3f(0.15, 0.15, 0.15)
            glLineWidth(0.8)
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, va)
            glDrawArrays(GL_TRIANGLES, 0, n)
            glDisableClientState(GL_VERTEX_ARRAY)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glDisable(GL_POLYGON_OFFSET_FILL)
            glLineWidth(1.0)
            glEnable(GL_LIGHTING)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self, mesh_data):
        """Display *mesh_data* (as returned by model_loader.load_model)."""
        self.mesh_data = mesh_data
        self._fit_model()
        self.update()

    def reset_view(self):
        if self.mesh_data:
            self._fit_model()
        else:
            self.target   = [0.0, 0.0, 0.0]
            self.radius   = 5.0
            self.azimuth  = 45.0
            self.elevation = 30.0
        self.update()

    def set_wireframe(self, enabled):
        self.show_wireframe = enabled
        self.show_overlay_wireframe = False
        self.update()

    def set_overlay_wireframe(self, enabled):
        self.show_wireframe = False
        self.show_overlay_wireframe = enabled
        self.update()

    def set_show_axes(self, enabled):
        self.show_axes = enabled
        self.update()

    def set_show_grid(self, enabled):
        self.show_grid = enabled
        self.update()

    def set_material(self, material):
        """Apply (or clear) a material.  Pass *None* to revert to default rendering."""
        self.makeCurrent()
        self._clear_material_textures()
        self._material = material
        if material is not None:
            self._upload_material_textures(material)
        self.doneCurrent()
        self.update()

    def get_material(self):
        """Return the currently active Material, or None."""
        return self._material

    # ------------------------------------------------------------------
    # Mouse / keyboard / drag-drop events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        self._last_mouse  = event.pos()
        self._mouse_button = event.button()

    def mouseMoveEvent(self, event):
        dx = event.x() - self._last_mouse.x()
        dy = event.y() - self._last_mouse.y()

        if self._mouse_button == Qt.LeftButton:
            self.azimuth   -= dx * 0.4
            self.elevation += dy * 0.4
            self.elevation  = max(-89.0, min(89.0, self.elevation))
        elif self._mouse_button in (Qt.RightButton, Qt.MiddleButton):
            self._pan(dx, dy)

        self._last_mouse = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        self._mouse_button = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 0.88 if delta > 0 else 1.0 / 0.88
        self.radius = max(0.001, self.radius * factor)
        self.update()

    def mouseDoubleClickEvent(self, event):
        self.reset_view()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.parent().window().open_file(path)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fit_model(self):
        if self.mesh_data is None:
            return
        bbox_min = self.mesh_data["bbox_min"]
        bbox_max = self.mesh_data["bbox_max"]
        center   = (bbox_min + bbox_max) / 2.0
        extent   = float(np.linalg.norm(bbox_max - bbox_min))
        if extent == 0:
            extent = 1.0
        self.target    = center.tolist()
        self.radius    = extent * 1.5
        self.azimuth   = 45.0
        self.elevation = 30.0

    def _pan(self, dx, dy):
        """Translate the camera target in screen-plane coordinates."""
        phi   = math.radians(self.azimuth)
        theta = math.radians(self.elevation)
        right = [ math.cos(phi),                          0.0, -math.sin(phi)]
        up    = [-math.sin(theta) * math.sin(phi),  math.cos(theta),
                 -math.sin(theta) * math.cos(phi)]
        speed = self.radius * 0.001
        for i in range(3):
            self.target[i] += (-dx * right[i] + dy * up[i]) * speed
