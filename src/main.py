"""Entry point for the 3D Model Viewer application."""

import os
import sys

# When running as a frozen PyInstaller bundle on Windows, add the temp
# extraction directory to the DLL search path so that native libraries
# (e.g. the bundled assimp DLL used by pyassimp) can be located.
if hasattr(sys, "_MEIPASS") and os.name == "nt":
    os.add_dll_directory(sys._MEIPASS)

# Ensure the src directory is on the path when running directly
sys.path.insert(0, os.path.dirname(__file__))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from mainwindow import MainWindow


def main():
    # High-DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("3D Model Viewer")
    app.setOrganizationName("3DModelViewer")

    window = MainWindow()
    window.show()

    # If a file was passed on the command line, open it
    if len(sys.argv) > 1:
        window.open_file(sys.argv[1])

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
