"""OpenGL 3-D viewer widget with orbit / pan / zoom controls."""

import math

import numpy as np
from OpenGL.GL import (
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
)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QOpenGLWidget


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
        # always orbits with elevation clamped to ±89° so this path is only reached
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
            self._draw_mesh()

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

    def _draw_mesh(self):
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
    # Public API
    # ------------------------------------------------------------------

    def load_model(self, mesh_data: dict):
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

    def set_wireframe(self, enabled: bool):
        self.show_wireframe = enabled
        self.show_overlay_wireframe = False
        self.update()

    def set_overlay_wireframe(self, enabled: bool):
        self.show_wireframe = False
        self.show_overlay_wireframe = enabled
        self.update()

    def set_show_axes(self, enabled: bool):
        self.show_axes = enabled
        self.update()

    def set_show_grid(self, enabled: bool):
        self.show_grid = enabled
        self.update()

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
