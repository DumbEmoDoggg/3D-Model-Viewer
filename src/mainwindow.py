"""Main application window."""

import os

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QToolBar,
)
from PyQt5.QtGui import QIcon, QKeySequence

from model_loader import FILE_FILTER, load_model
from viewer_widget import ViewerWidget


# ---------------------------------------------------------------------------
# Background loader thread
# ---------------------------------------------------------------------------

class _LoadThread(QThread):
    """Loads a 3D model file on a background thread."""

    loaded  = pyqtSignal(dict)
    errored = pyqtSignal(str)

    def __init__(self, filepath: str, parent=None):
        super().__init__(parent)
        self._filepath = filepath

    def run(self):
        try:
            data = load_model(self._filepath)
            self.loaded.emit(data)
        except Exception as exc:  # broad catch intentional: trimesh raises diverse error types
            self.errored.emit(str(exc))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Top-level window for the 3D model viewer."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Model Viewer")
        self.resize(1100, 750)

        self._viewer = ViewerWidget(self)
        self.setCentralWidget(self._viewer)

        self._build_menus()
        self._build_toolbar()
        self._build_statusbar()

        self._load_thread = None

        # Accept file drops on the main window too
        self.setAcceptDrops(True)

    # ------------------------------------------------------------------
    # Menu / toolbar / status bar construction
    # ------------------------------------------------------------------

    def _build_menus(self):
        menu = self.menuBar()

        # File menu
        file_menu = menu.addMenu("&File")

        open_act = QAction("&Open…", self)
        open_act.setShortcut(QKeySequence.Open)
        open_act.setStatusTip("Open a 3D model file")
        open_act.triggered.connect(self._on_open)
        file_menu.addAction(open_act)

        file_menu.addSeparator()

        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence.Quit)
        quit_act.setStatusTip("Quit the application")
        quit_act.triggered.connect(QApplication.quit)
        file_menu.addAction(quit_act)

        # View menu
        view_menu = menu.addMenu("&View")

        reset_act = QAction("&Reset View", self)
        reset_act.setShortcut("R")
        reset_act.setStatusTip("Reset camera to fit model")
        reset_act.triggered.connect(self._viewer.reset_view)
        view_menu.addAction(reset_act)

        view_menu.addSeparator()

        # Shading radio group
        shading_group = QActionGroup(self)
        shading_group.setExclusive(True)

        self._solid_act = QAction("&Solid", self)
        self._solid_act.setCheckable(True)
        self._solid_act.setChecked(True)
        self._solid_act.triggered.connect(self._on_solid)
        shading_group.addAction(self._solid_act)
        view_menu.addAction(self._solid_act)

        self._wire_act = QAction("&Wireframe", self)
        self._wire_act.setCheckable(True)
        self._wire_act.triggered.connect(self._on_wireframe)
        shading_group.addAction(self._wire_act)
        view_menu.addAction(self._wire_act)

        self._overlay_act = QAction("Solid + &Wireframe", self)
        self._overlay_act.setCheckable(True)
        self._overlay_act.triggered.connect(self._on_overlay)
        shading_group.addAction(self._overlay_act)
        view_menu.addAction(self._overlay_act)

        view_menu.addSeparator()

        self._axes_act = QAction("Show &Axes", self)
        self._axes_act.setCheckable(True)
        self._axes_act.setChecked(True)
        self._axes_act.triggered.connect(
            lambda checked: self._viewer.set_show_axes(checked)
        )
        view_menu.addAction(self._axes_act)

        self._grid_act = QAction("Show &Grid", self)
        self._grid_act.setCheckable(True)
        self._grid_act.setChecked(True)
        self._grid_act.triggered.connect(
            lambda checked: self._viewer.set_show_grid(checked)
        )
        view_menu.addAction(self._grid_act)

        # Help menu
        help_menu = menu.addMenu("&Help")
        about_act = QAction("&About", self)
        about_act.triggered.connect(self._on_about)
        help_menu.addAction(about_act)

    def _build_toolbar(self):
        tb = QToolBar("Main Toolbar", self)
        tb.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, tb)

        open_btn = QAction("Open", self)
        open_btn.setStatusTip("Open a 3D model file")
        open_btn.triggered.connect(self._on_open)
        tb.addAction(open_btn)

        tb.addSeparator()

        reset_btn = QAction("Reset View", self)
        reset_btn.setStatusTip("Reset camera to fit model (R)")
        reset_btn.triggered.connect(self._viewer.reset_view)
        tb.addAction(reset_btn)

        tb.addSeparator()

        solid_btn = QAction("Solid", self)
        solid_btn.setStatusTip("Render solid shading")
        solid_btn.triggered.connect(self._on_solid)
        tb.addAction(solid_btn)

        wire_btn = QAction("Wireframe", self)
        wire_btn.setStatusTip("Render wireframe")
        wire_btn.triggered.connect(self._on_wireframe)
        tb.addAction(wire_btn)

        overlay_btn = QAction("Solid+Wire", self)
        overlay_btn.setStatusTip("Render solid with wireframe overlay")
        overlay_btn.triggered.connect(self._on_overlay)
        tb.addAction(overlay_btn)

    def _build_statusbar(self):
        self._status_file  = QLabel("No file loaded")
        self._status_stats = QLabel("")
        self.statusBar().addWidget(self._status_file, 1)
        self.statusBar().addPermanentWidget(self._status_stats)

    # ------------------------------------------------------------------
    # Shading helpers
    # ------------------------------------------------------------------

    def _on_solid(self):
        self._solid_act.setChecked(True)
        self._viewer.show_wireframe = False
        self._viewer.show_overlay_wireframe = False
        self._viewer.update()

    def _on_wireframe(self):
        self._wire_act.setChecked(True)
        self._viewer.set_wireframe(True)

    def _on_overlay(self):
        self._overlay_act.setChecked(True)
        self._viewer.set_overlay_wireframe(True)

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open 3D Model", "", FILE_FILTER
        )
        if path:
            self.open_file(path)

    def open_file(self, path: str):
        """Load *path* asynchronously and display it."""
        if self._load_thread and self._load_thread.isRunning():
            return  # Ignore concurrent requests

        self._status_file.setText(f"Loading {os.path.basename(path)}…")
        self._status_stats.setText("")

        progress = QProgressDialog(
            f"Loading {os.path.basename(path)}…", None, 0, 0, self
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(300)

        self._load_thread = _LoadThread(path, self)

        def on_loaded(data):
            progress.close()
            self._viewer.load_model(data)
            fname = os.path.basename(data["filepath"])
            self._status_file.setText(fname)
            self._status_stats.setText(
                f"Vertices: {data['vertex_count']:,}   "
                f"Faces: {data['face_count']:,}"
            )

        def on_error(msg):
            progress.close()
            self._status_file.setText("Load failed")
            QMessageBox.critical(self, "Load Error", f"Failed to load model:\n{msg}")

        self._load_thread.loaded.connect(on_loaded)
        self._load_thread.errored.connect(on_error)
        self._load_thread.finished.connect(progress.close)
        self._load_thread.start()

    # ------------------------------------------------------------------
    # About dialog
    # ------------------------------------------------------------------

    def _on_about(self):
        QMessageBox.about(
            self,
            "About 3D Model Viewer",
            "<h2>3D Model Viewer</h2>"
            "<p>A cross-platform desktop viewer for common 3D model formats.</p>"
            "<p><b>Supported formats:</b> OBJ, STL, PLY, GLTF/GLB, 3MF, OFF, DAE</p>"
            "<p><b>Controls:</b><br>"
            "Left drag — orbit<br>"
            "Right drag — pan<br>"
            "Scroll wheel — zoom<br>"
            "Double-click — reset view<br>"
            "R — reset view</p>"
            "<p>Built with Python, PyQt5, PyOpenGL, and trimesh.</p>",
        )

    # ------------------------------------------------------------------
    # Drag-and-drop on the window
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            self.open_file(urls[0].toLocalFile())

    # ------------------------------------------------------------------
    # Key shortcuts
    # ------------------------------------------------------------------

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_R:
            self._viewer.reset_view()
        elif key == Qt.Key_W:
            self._on_wireframe()
        elif key == Qt.Key_S:
            self._on_solid()
        elif key == Qt.Key_O:
            self._on_overlay()
        else:
            super().keyPressEvent(event)
