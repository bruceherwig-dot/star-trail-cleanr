#!/usr/bin/env python3
"""
Auto-discovers all installed packages that contain data files and runs PyInstaller.
Prevents missing-data-file crashes without requiring a manually maintained list.
"""
import os, site, subprocess, sys

sep = ';' if sys.platform == 'win32' else ':'

# Extensions PyInstaller handles natively — exclude from --add-data
SKIP_EXT = {'.py', '.pyc', '.pyo', '.pyd', '.so', '.dylib', '.dll'}

# Packages that must NEVER be bundled, even if they're present in the local
# site-packages. These are NOT runtime deps of Star Trail CleanR. The GitHub
# CI environment never installs them, so excluding them is a no-op on CI and
# prevents local-environment pollution (especially PyQt5, which conflicts
# with PySide6 at runtime if both are bundled).
SKIP_PACKAGES = {
    'PyQt5', 'PyQt6', 'PySide2',   # competing Qt bindings — we use PySide6
    'gradio', 'gradio_client',      # old v0.x GUI lib, replaced by PySide6
    'borb',                         # PDF lib, not used
    'transformers', 'tokenizers',   # Hugging Face, not used at runtime
    'astropy',                      # wrong astronomy library, not used
    'onnxruntime',                  # optional ultralytics export format, not runtime
    'tensorboard', 'tensorboardX',  # training-time only
    'grpc',                         # not used
    'polars', '_polars_runtime_32', '_polars_runtime_64',  # DataFrame lib, not used
    # NOTE: sympy is a torch runtime dep via torch._dynamo — do NOT skip
    'streamlit',                    # alternative GUI, not used
    'flask', 'fastapi',             # web frameworks, not used
    'jupyter', 'ipykernel', 'ipython', 'notebook',  # notebook stack, not used
}

site_dirs = []
try:
    site_dirs += site.getsitepackages()
except Exception:
    pass
try:
    ud = site.getusersitepackages()
    if ud not in site_dirs:
        site_dirs.append(ud)
except Exception:
    pass

# Always include the algorithm script and version file
add_data = [f'astro_clean_v5.py{sep}.', f'version.txt{sep}.',
            f'modules{sep}modules', f'assets{sep}assets']
seen = set()

for site_dir in site_dirs:
    if not os.path.isdir(site_dir):
        continue
    for pkg_name in sorted(os.listdir(site_dir)):
        if pkg_name in seen:
            continue
        if pkg_name in SKIP_PACKAGES:
            continue
        if pkg_name.endswith(('.dist-info', '.egg-info', '.egg-link', '__pycache__')):
            continue
        pkg_dir = os.path.join(site_dir, pkg_name)
        if not os.path.isdir(pkg_dir):
            continue
        # Walk package directory looking for any data file
        for root, dirs, files in os.walk(pkg_dir):
            for f in files:
                if os.path.splitext(f)[1] not in SKIP_EXT:
                    add_data.append(f'{pkg_dir}{sep}{pkg_name}')
                    seen.add(pkg_name)
                    break
            if pkg_name in seen:
                break

icon_ext = '.ico' if sys.platform == 'win32' else '.icns'
icon_path = os.path.join(os.path.dirname(__file__), 'assets', 'StarTrailCleanR' + icon_ext)

cmd = [
    sys.executable, '-m', 'PyInstaller',
    '--onedir', '--windowed', '--noupx', 'star_trail_cleanr.py',
    '--name', 'StarTrailCleanR',
    '--icon', icon_path,
    '--collect-all', 'cv2',
    '--collect-all', 'numpy',
    '--collect-all', 'PySide6',
    '--collect-all', 'sahi',
    '--collect-all', 'ultralytics',
]
# Force PyInstaller to exclude the same skip list at the module-analysis level,
# not just the data-file walker. This stops transitive imports from pulling
# them back in.
for pkg in sorted(SKIP_PACKAGES):
    cmd += ['--exclude-module', pkg]
for d in add_data:
    cmd += ['--add-data', d]

# Bundle the YOLO model so the frozen app doesn't depend on a local path
model_pt = os.path.join(os.path.dirname(__file__), 'assets', 'best.pt')
if os.path.isfile(model_pt):
    cmd += ['--add-data', model_pt + sep + '.']
    print(f'Bundling YOLO model: {model_pt}')
else:
    print(f'WARNING: YOLO model not found at {model_pt} — build will lack model')

print(f'Bundling data from {len(seen)} packages:')
for pkg in sorted(seen):
    print(f'  {pkg}')

result = subprocess.run(cmd)
if result.returncode != 0:
    sys.exit(result.returncode)

import shutil

def dir_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / 1024 / 1024

if sys.platform == 'darwin':
    dist_root = os.path.join('dist', 'StarTrailCleanR.app')
else:
    dist_root = os.path.join('dist', 'StarTrailCleanR')

print(f'\nPost-build cleanup, walking: {dist_root}')
before = dir_size_mb(os.path.join('dist'))
print(f'Before cleanup: {before:.1f} MB')

# Ground-truth diagnostic: print every torch/ and ultralytics/ directory found,
# so the build log shows the real layout even if cleanup misses something.
print('\nDiagnostic: torch/ and ultralytics/ directories found:')
for pkg in ('torch', 'ultralytics'):
    for root, dirs, _ in os.walk(dist_root):
        if pkg in dirs:
            full = os.path.join(root, pkg)
            size = dir_size_mb(full)
            rel = os.path.relpath(full, 'dist')
            print(f'  {rel}  ({size:.1f} MB)')

CLEANUP_PATHS = [
    ('torch', 'include'),
    ('torch', 'test'),
    ('torch', 'testing'),
    ('ultralytics', 'assets'),
    ('ultralytics', 'cfg', 'datasets'),
    # PySide6 Qt frameworks not used by a widget-only app. Biggest target is
    # QtWebEngineCore (full Chromium engine, ~280 MB uncompressed on Mac).
    ('Qt', 'lib', 'QtWebEngineCore.framework'),
    ('Qt', 'lib', 'QtWebEngineQuick.framework'),
    ('Qt', 'lib', 'QtWebEngineWidgets.framework'),
    ('Qt', 'lib', 'QtQuick.framework'),
    ('Qt', 'lib', 'QtQuick3D.framework'),
    ('Qt', 'lib', 'QtQuick3DRuntimeRender.framework'),
    ('Qt', 'lib', 'QtQuickControls2.framework'),
    ('Qt', 'lib', 'QtQuickControls2Imagine.framework'),
    ('Qt', 'lib', 'QtQuickDialogs2.framework'),
    ('Qt', 'lib', 'QtQuickDialogs2QuickImpl.framework'),
    ('Qt', 'lib', 'QtQml.framework'),
    ('Qt', 'lib', 'QtQmlCompiler.framework'),
    ('Qt', 'lib', 'QtQmlModels.framework'),
    ('Qt', 'lib', 'QtQmlWorkerScript.framework'),
    ('Qt', 'lib', 'QtDesigner.framework'),
    ('Qt', 'lib', 'QtDesignerComponents.framework'),
    ('Qt', 'lib', 'QtShaderTools.framework'),
    ('Qt', 'lib', 'QtPdf.framework'),
    ('Qt', 'lib', 'Qt3DCore.framework'),
    ('Qt', 'lib', 'Qt3DRender.framework'),
    ('Qt', 'lib', 'Qt3DAnimation.framework'),
    ('Qt', 'lib', 'Qt3DExtras.framework'),
    ('Qt', 'lib', 'Qt3DInput.framework'),
    ('Qt', 'lib', 'Qt3DLogic.framework'),
    ('Qt', 'lib', 'Qt3DQuick.framework'),
    ('Qt', 'lib', 'Qt3DQuickAnimation.framework'),
    ('Qt', 'lib', 'Qt3DQuickExtras.framework'),
    ('Qt', 'lib', 'Qt3DQuickInput.framework'),
    ('Qt', 'lib', 'Qt3DQuickRender.framework'),
    ('Qt', 'lib', 'Qt3DQuickScene2D.framework'),
    ('Qt', 'lib', 'QtGraphs.framework'),
    ('Qt', 'lib', 'QtCharts.framework'),
    ('Qt', 'lib', 'QtDataVisualization.framework'),
    ('Qt', 'lib', 'QtMultimedia.framework'),
    ('Qt', 'lib', 'QtMultimediaWidgets.framework'),
    ('Qt', 'lib', 'QtMultimediaQuick.framework'),
    ('Qt', 'lib', 'QtVirtualKeyboard.framework'),
    ('Qt', 'lib', 'QtWebChannel.framework'),
    ('Qt', 'lib', 'QtWebSockets.framework'),
    ('Qt', 'lib', 'QtWebView.framework'),
    ('Qt', 'lib', 'QtLocation.framework'),
    ('Qt', 'lib', 'QtPositioning.framework'),
    ('Qt', 'lib', 'QtBluetooth.framework'),
    ('Qt', 'lib', 'QtNfc.framework'),
    ('Qt', 'lib', 'QtSensors.framework'),
    ('Qt', 'lib', 'QtSerialBus.framework'),
    ('Qt', 'lib', 'QtSerialPort.framework'),
    ('Qt', 'lib', 'QtRemoteObjects.framework'),
    ('Qt', 'lib', 'QtTextToSpeech.framework'),
    ('Qt', 'lib', 'QtSpatialAudio.framework'),
    ('Qt', 'lib', 'QtTest.framework'),
    # Windows equivalents (DLLs, not frameworks). Handled by a separate pass below.
]

removed = []
for root, dirs, _ in os.walk(dist_root):
    for d in list(dirs):
        full = os.path.join(root, d)
        rel_parts = os.path.relpath(full, dist_root).split(os.sep)
        for pattern in CLEANUP_PATHS:
            if len(rel_parts) >= len(pattern) and tuple(rel_parts[-len(pattern):]) == pattern:
                size = dir_size_mb(full)
                shutil.rmtree(full, ignore_errors=True)
                removed.append((os.path.relpath(full, 'dist'), size))
                dirs.remove(d)
                break

print('\nCleanup removed:')
if not removed:
    print('  (nothing matched — check the diagnostic above)')
for path, size in removed:
    print(f'  {path}  ({size:.1f} MB)')

after = dir_size_mb(os.path.join('dist'))
print(f'\nAfter cleanup: {after:.1f} MB  (saved {before - after:.1f} MB)')

sys.exit(0)
