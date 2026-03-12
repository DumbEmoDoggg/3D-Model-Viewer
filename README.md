# 3D Model Viewer

A cross-platform desktop application for viewing common 3D model formats, built with Python, PyQt5, PyOpenGL, and trimesh.

## Features

- **Supported formats**: OBJ, STL, PLY, GLTF/GLB, 3MF, OFF, DAE
- **Orbit / Pan / Zoom** camera controls
- **Shading modes**: Solid, Wireframe, Solid + Wireframe overlay
- Toggleable **grid** and **XYZ axes**
- Drag-and-drop file loading
- Command-line file argument support

## Controls

| Action | Input |
|--------|-------|
| Orbit  | Left mouse drag |
| Pan    | Right mouse drag |
| Zoom   | Scroll wheel |
| Reset view | Double-click or press **R** |
| Open file | Ctrl+O |
| Solid shading | **S** |
| Wireframe | **W** |
| Solid + Wire overlay | **O** |

## Running from source

```bash
# Install dependencies
pip install -r requirements.txt

# Launch
python src/main.py

# Open a file directly
python src/main.py model.stl
```

## Building a standalone executable

### Linux

```bash
pip install pyinstaller
pyinstaller --onefile --name 3d-model-viewer src/main.py
# executable at dist/3d-model-viewer
```

### Windows

```bat
pip install pyinstaller
pyinstaller --onefile --windowed --name 3D-Model-Viewer src\main.py
# executable at dist\3D-Model-Viewer.exe
```

## CI / Automated builds

GitHub Actions workflows build and upload ready-to-run executables on every push:

| Workflow | Runner | Artifact |
|----------|--------|----------|
| [Build Linux](.github/workflows/build-linux.yml) | `ubuntu-latest` | `3D-Model-Viewer-Linux` |
| [Build Windows](.github/workflows/build-windows.yml) | `windows-latest` | `3D-Model-Viewer-Windows` |

Download the latest artifact from the **Actions** tab in GitHub.

## Dependencies

| Package | Purpose |
|---------|---------|
| PyQt5 | GUI framework and OpenGL context |
| PyOpenGL | OpenGL bindings |
| numpy | Vertex array math |
| trimesh | 3D model loading (OBJ, STL, PLY, GLTF, …) |