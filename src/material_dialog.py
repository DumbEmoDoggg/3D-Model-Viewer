"""Dialog for configuring a PBR-lite material."""

import os
import copy

from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QColorDialog,
)
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtCore import Qt

from material import Material

# Common image file filter
_IMAGE_FILTER = (
    "Images (*.png *.jpg *.jpeg *.bmp *.tga *.tiff *.gif *.hdr);;"
    "All Files (*)"
)


def _color_pixmap(r: float, g: float, b: float, size: int = 20) -> QPixmap:
    """Return a solid-colour QPixmap for use on a button."""
    px = QPixmap(size, size)
    px.fill(QColor(int(r * 255), int(g * 255), int(b * 255)))
    return px


class _TextureRow(QHBoxLayout):
    """A row with a browse button, path label, and clear button."""

    def __init__(self, parent_dialog: QDialog, initial_path: str | None = None):
        super().__init__()
        self._parent = parent_dialog

        self._path_edit = QLineEdit()
        self._path_edit.setReadOnly(True)
        self._path_edit.setPlaceholderText("(no texture – using value)")
        self._path_edit.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        if initial_path:
            self._path_edit.setText(initial_path)

        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse)

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(55)
        clear_btn.clicked.connect(self._clear)

        self.addWidget(self._path_edit)
        self.addWidget(browse_btn)
        self.addWidget(clear_btn)

    # ------------------------------------------------------------------
    def _browse(self):
        start = (
            os.path.dirname(self._path_edit.text())
            if self._path_edit.text()
            else ""
        )
        path, _ = QFileDialog.getOpenFileName(
            self._parent, "Select Texture Image", start, _IMAGE_FILTER
        )
        if path:
            self._path_edit.setText(path)

    def _clear(self):
        self._path_edit.clear()

    # ------------------------------------------------------------------
    @property
    def path(self) -> str | None:
        t = self._path_edit.text().strip()
        return t if t else None


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

class MaterialDialog(QDialog):
    """Modal dialog that lets the user configure a PBR-lite material.

    Call :meth:`get_material` after :meth:`exec_` to retrieve the result.
    """

    def __init__(self, parent=None, material: Material | None = None):
        super().__init__(parent)
        self.setWindowTitle("Set Material")
        self.setMinimumWidth(500)

        # Work on a copy so Cancel doesn't mutate the original
        self._mat = copy.deepcopy(material) if material else Material()
        self._result: Material | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_base_color_group())
        layout.addWidget(self._build_normal_map_group())
        layout.addWidget(self._build_smoothness_group())
        layout.addWidget(self._build_metallic_group())

        info = QLabel(
            "<i>Textures require the model to have UV coordinates.<br>"
            "Smoothness and Metallic maps use the red channel.</i>"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Group builders
    # ------------------------------------------------------------------

    def _build_base_color_group(self) -> QGroupBox:
        grp = QGroupBox("Base Color")
        form = QFormLayout(grp)

        # Colour picker button
        r, g, b, _ = self._mat.base_color
        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(40, 24)
        self._color_btn.setIcon(self.style().standardIcon(0))  # placeholder
        self._color_btn.setIconSize(self._color_btn.size())
        self._color_btn.setToolTip("Click to choose base colour")
        self._update_color_btn(r, g, b)
        self._color_btn.clicked.connect(self._pick_color)

        form.addRow("Colour:", self._color_btn)

        self._base_tex = _TextureRow(self, self._mat.base_color_texture)
        form.addRow("Texture:", self._base_tex)
        return grp

    def _build_normal_map_group(self) -> QGroupBox:
        grp = QGroupBox("Normal Map")
        form = QFormLayout(grp)
        self._normal_tex = _TextureRow(self, self._mat.normal_map_texture)
        form.addRow("Texture:", self._normal_tex)
        return grp

    def _build_smoothness_group(self) -> QGroupBox:
        grp = QGroupBox("Smoothness")
        form = QFormLayout(grp)

        self._smooth_spin = QDoubleSpinBox()
        self._smooth_spin.setRange(0.0, 1.0)
        self._smooth_spin.setSingleStep(0.05)
        self._smooth_spin.setDecimals(2)
        self._smooth_spin.setValue(self._mat.smoothness)
        form.addRow("Value (0–1):", self._smooth_spin)

        self._smooth_tex = _TextureRow(self, self._mat.smoothness_texture)
        form.addRow("Texture:", self._smooth_tex)
        return grp

    def _build_metallic_group(self) -> QGroupBox:
        grp = QGroupBox("Metallic")
        form = QFormLayout(grp)

        self._metal_spin = QDoubleSpinBox()
        self._metal_spin.setRange(0.0, 1.0)
        self._metal_spin.setSingleStep(0.05)
        self._metal_spin.setDecimals(2)
        self._metal_spin.setValue(self._mat.metallic)
        form.addRow("Value (0–1):", self._metal_spin)

        self._metal_tex = _TextureRow(self, self._mat.metallic_texture)
        form.addRow("Texture:", self._metal_tex)
        return grp

    # ------------------------------------------------------------------
    # Colour picker
    # ------------------------------------------------------------------

    def _pick_color(self):
        r, g, b, a = self._mat.base_color
        initial = QColor(int(r * 255), int(g * 255), int(b * 255))
        col = QColorDialog.getColor(initial, self, "Choose Base Colour")
        if col.isValid():
            self._mat.base_color = (
                col.redF(), col.greenF(), col.blueF(),
                self._mat.base_color[3],
            )
            self._update_color_btn(col.redF(), col.greenF(), col.blueF())

    def _update_color_btn(self, r: float, g: float, b: float):
        px = _color_pixmap(r, g, b, 24)
        self._color_btn.setIcon(self.style().standardIcon(0))  # clear first
        self._color_btn.setIcon(px if not px.isNull() else self.style().standardIcon(0))
        # Use stylesheet as a fallback that always works
        self._color_btn.setStyleSheet(
            f"background-color: rgb({int(r*255)},{int(g*255)},{int(b*255)});"
        )

    # ------------------------------------------------------------------
    # Accept / retrieve result
    # ------------------------------------------------------------------

    def _on_accept(self):
        self._mat.base_color_texture = self._base_tex.path
        self._mat.normal_map_texture = self._normal_tex.path
        self._mat.smoothness = self._smooth_spin.value()
        self._mat.smoothness_texture = self._smooth_tex.path
        self._mat.metallic = self._metal_spin.value()
        self._mat.metallic_texture = self._metal_tex.path
        self._result = self._mat
        self.accept()

    def get_material(self) -> Material | None:
        """Return the configured :class:`Material`, or *None* if cancelled."""
        return self._result
