#!/usr/bin/env python3
"""
Auto-discovers all installed packages that contain data files and runs PyInstaller.
Prevents missing-data-file crashes without requiring a manually maintained list.
"""
import os, site, subprocess, sys

# Reproducible builds: pin a deterministic timestamp before PyInstaller starts.
# Python's bytecode compiler embeds the source file's mtime in every .pyc; on
# CI runners every fresh checkout gives source files a brand-new mtime, so the
# same source produces different .pyc bytes between builds. Setting
# SOURCE_DATE_EPOCH overrides that mtime with a fixed value, making .pyc
# output deterministic. This is what cuts Sparkle/WinSparkle delta updates
# from ~237 MB (v1.98 → v1.99 measured 2026-04-30) toward the 30-50 MB range.
# Value is the standard "reproducible-builds-friendly" timestamp (2020-01-01);
# the exact number is irrelevant as long as it never changes.
os.environ['SOURCE_DATE_EPOCH'] = '1577836800'

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
    # 2026-04-30 audit additions — none of these are imported by STC's code.
    # All transitive: SAHI's optional VLM integrations (openai, anthropic),
    # ultralytics' training plots (matplotlib, pandas), labelme tooling
    # (lxml, imgviz, labelme). Verified via grep across modules/ + top-level
    # *.py — zero direct imports.
    'pandas',                       # ultralytics training output, not runtime
    'matplotlib',                   # ultralytics plots, not runtime
    'lxml',                         # labelme XML, not used at runtime
    'openai',                       # sahi optional VLM detector, not used
    'anthropic',                    # sahi optional VLM detector, not used
    'imgviz',                       # labelme dep, not used at runtime
    'labelme',                      # annotation tool, not runtime
    # 2026-05-01 audit additions. Verified safe to exclude:
    #   pip: not imported by our code; bundled by accident
    #   astropy_iers_data: required only by astropy (already excluded)
    #   fontTools: required only by borb + matplotlib (both already excluded)
    'pip',                          # package installer, not needed at runtime
    'astropy_iers_data',            # orphan from astropy exclusion
    'fontTools',                    # orphan from matplotlib + borb exclusions
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

if sys.platform == 'win32':
    icon_ext = '.ico'
elif sys.platform == 'darwin':
    icon_ext = '.icns'
else:
    icon_ext = '.png'
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
    '--collect-all', 'skimage',
    '--collect-all', 'tifffile',
    '--runtime-hook', 'rthooks/pyi_rthook_gpu_override.py',
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
    # Qt Multimedia plugin loaders. Parent framework is removed above; these
    # plugins (ffmpeg backend + macOS-native AVFoundation backend) become
    # orphaned and are the only callers of Qt's bundled ffmpeg libs.
    ('Qt', 'plugins', 'multimedia'),
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

# File-level cleanup inside torch/lib/. Targets CUDA libraries we do not use:
#  - NCCL: NVIDIA's multi-GPU communication library. Star Trail CleanR runs
#    single-GPU inference, so NCCL is never loaded.
#  - nvJPEG: NVIDIA's GPU-side JPEG reader/writer. We read images via OpenCV
#    on the CPU, so nvJPEG is never loaded.
# Applies to both CPU and CUDA builds; on CPU-only wheels these files do not
# exist, and the cleanup is a no-op. On CUDA (NVIDIA) builds, removes them
# preemptively to trim the installer.
CUDA_LIB_PREFIXES_TO_REMOVE = ('libnccl', 'nccl', 'libnvjpeg', 'nvjpeg')
removed_files = []
for root, dirs, files in os.walk(dist_root):
    if os.path.basename(root) != 'lib':
        continue
    if os.path.basename(os.path.dirname(root)) != 'torch':
        continue
    for f in list(files):
        for prefix in CUDA_LIB_PREFIXES_TO_REMOVE:
            if f.startswith(prefix):
                full_file = os.path.join(root, f)
                try:
                    fsize = os.path.getsize(full_file) / 1024 / 1024
                    os.remove(full_file)
                    removed_files.append((os.path.relpath(full_file, 'dist'), fsize))
                except OSError:
                    pass
                break

if removed_files:
    print('\nCUDA-specific library cleanup (NCCL, nvJPEG):')
    for path, fsize in removed_files:
        print(f'  {path}  ({fsize:.1f} MB)')
else:
    print('\nCUDA-specific library cleanup: nothing matched (expected for CPU-only builds).')

# PySide6 ships Qt SDK build utilities (Assistant.app, Linguist.app, Designer.app,
# qmlls, qmlformat, qmllint, balsam, balsamui, lrelease, lupdate, qsb, svgtoqml).
# Never imported at runtime. PyInstaller renames .app -> __dot__app on Mac.
PYSIDE_DEV_TOOL_NAMES = {
    'Assistant__dot__app', 'Linguist__dot__app', 'Designer__dot__app',
    'assistant.exe', 'linguist.exe', 'designer.exe',
    'qmlls', 'qmlformat', 'qmllint',
    'balsam', 'balsamui',
    'lrelease', 'lupdate',
    'qsb', 'svgtoqml',
    'qmlls.exe', 'qmlformat.exe', 'qmllint.exe',
    'balsam.exe', 'balsamui.exe',
    'lrelease.exe', 'lupdate.exe',
    'qsb.exe', 'svgtoqml.exe',
}

devtool_removed = []
for root, dirs, files in os.walk(dist_root):
    if 'PySide6' not in root.split(os.sep):
        continue
    for f in list(files):
        if f in PYSIDE_DEV_TOOL_NAMES:
            full = os.path.join(root, f)
            try:
                fsize = os.path.getsize(full) / 1024 / 1024
                os.remove(full)
                devtool_removed.append((os.path.relpath(full, 'dist'), fsize))
            except OSError:
                pass
    for d in list(dirs):
        if d in PYSIDE_DEV_TOOL_NAMES:
            full = os.path.join(root, d)
            dsize = dir_size_mb(full)
            shutil.rmtree(full, ignore_errors=True)
            devtool_removed.append((os.path.relpath(full, 'dist'), dsize))
            dirs.remove(d)

if devtool_removed:
    print('\nPySide6 dev-tool cleanup:')
    for path, fsize in devtool_removed:
        print(f'  {path}  ({fsize:.1f} MB)')
else:
    print('\nPySide6 dev-tool cleanup: nothing matched.')

# Qt-bundled ffmpeg lib cleanup. Qt Multimedia framework + plugins are removed
# above; the ffmpeg-family codec libs in PySide6/Qt/lib (Mac) or Qt/bin
# (Windows) become orphaned — nothing else in the bundle links them. cv2 has
# its own ffmpeg copies under cv2/__dot__dylibs/ which we leave alone.
QT_FFMPEG_LIB_PREFIXES = (
    'libavcodec.', 'libavformat.', 'libavutil.',
    'libswscale.', 'libswresample.',
    'avcodec-', 'avformat-', 'avutil-',
    'swscale-', 'swresample-',
)
ffmpeg_removed = []
for root, dirs, files in os.walk(dist_root):
    if 'PySide6' not in root.split(os.sep):
        continue
    for f in list(files):
        for prefix in QT_FFMPEG_LIB_PREFIXES:
            if f.startswith(prefix):
                full_file = os.path.join(root, f)
                try:
                    fsize = os.path.getsize(full_file) / 1024 / 1024
                    os.remove(full_file)
                    ffmpeg_removed.append((os.path.relpath(full_file, 'dist'), fsize))
                except OSError:
                    pass
                break

if ffmpeg_removed:
    print('\nQt-bundled ffmpeg lib cleanup:')
    for path, fsize in ffmpeg_removed:
        print(f'  {path}  ({fsize:.1f} MB)')
else:
    print('\nQt-bundled ffmpeg lib cleanup: nothing matched.')

# Windows Qt-DLL cleanup — mirrors the Mac framework cleanup above.
# CLEANUP_PATHS removes ~30 unused Qt frameworks from the Mac .app bundle
# (QtWebEngineCore, Qt3DCore, QtMultimedia, etc.). The same modules ship
# on Windows as DLLs and .pyd Python bindings, but the Mac pass cannot see
# them because it matches on directory paths ending in `.framework`. This
# pass extracts the module names from CLEANUP_PATHS and removes the
# Windows-equivalent files. Without this, Windows installers ship the
# entire unused-Qt payload — historically responsible for our Windows
# bundles being ~200 MB heavier than they need to be.
if sys.platform == 'win32':
    qt_module_names = set()
    for pattern in CLEANUP_PATHS:
        last = pattern[-1]
        if last.endswith('.framework') and last.startswith('Qt'):
            # 'QtWebEngineCore.framework' -> 'WebEngineCore'
            qt_module_names.add(last[len('Qt'):-len('.framework')])

    win_removed = []
    for root, dirs, files in os.walk(dist_root):
        for f in list(files):
            for mod in qt_module_names:
                # Match: Qt6<mod>.dll (release), Qt6<mod>d.dll (debug),
                # <mod>.pyd (Python binding), Qt<mod>.pyd (alt naming).
                if f in (f'Qt6{mod}.dll', f'Qt6{mod}d.dll',
                         f'{mod}.pyd', f'Qt{mod}.pyd'):
                    full_file = os.path.join(root, f)
                    try:
                        fsize = os.path.getsize(full_file) / 1024 / 1024
                        os.remove(full_file)
                        win_removed.append((os.path.relpath(full_file, 'dist'), fsize))
                    except OSError:
                        pass
                    break

    if win_removed:
        print('\nWindows Qt-DLL cleanup (mirroring Mac framework cleanup):')
        for path, fsize in win_removed:
            print(f'  {path}  ({fsize:.1f} MB)')
    else:
        print('\nWindows Qt-DLL cleanup: nothing matched. '
              'Check Qt module naming if this is unexpected.')

# ── Sparkle / WinSparkle integration ────────────────────────────────────
# Copy the vendored auto-update framework into the bundle and inject the
# config keys Sparkle needs. WinSparkle on Windows reads its config at
# runtime via the ctypes wrapper in modules/winsparkle_updater.py — no
# Info.plist equivalent needed; just the DLL placement.

# GitHub Pages hosts the appcast XML feeds (one per platform) on the
# repo's gh-pages branch. URLs are stable across releases; only the
# advertised version inside the XML changes.
APPCAST_BASE = 'https://bruceherwig-dot.github.io/star-trail-cleanr'

sparkle_pubkey_path = os.path.join(os.path.dirname(__file__), 'assets', 'sparkle_public_key.txt')
sparkle_pubkey = None
if os.path.isfile(sparkle_pubkey_path):
    with open(sparkle_pubkey_path) as f:
        sparkle_pubkey = f.read().strip()

if sys.platform == 'darwin':
    import platform as _plat
    arch = 'apple-silicon' if _plat.machine() == 'arm64' else 'intel'
    appcast_url = f'{APPCAST_BASE}/appcast-mac-{arch}.xml'

    # Step 1: copy Sparkle.framework into the bundle. Use ditto, not cp -R
    # — ditto preserves Versions/A symlinks and code-signing seals; cp -R
    # corrupts both. (Per fman blog post on PyInstaller + Sparkle.)
    sparkle_src = os.path.join(os.path.dirname(__file__), 'vendored', 'Sparkle.framework')
    sparkle_dest = os.path.join(dist_root, 'Contents', 'Frameworks', 'Sparkle.framework')
    if os.path.isdir(sparkle_src):
        os.makedirs(os.path.dirname(sparkle_dest), exist_ok=True)
        if os.path.exists(sparkle_dest):
            shutil.rmtree(sparkle_dest, ignore_errors=True)
        result = subprocess.run(['ditto', sparkle_src, sparkle_dest], capture_output=True)
        if result.returncode == 0:
            sz = dir_size_mb(sparkle_dest)
            print(f'\nSparkle.framework copied into bundle ({sz:.1f} MB)')
        else:
            print(f'\nWARNING: ditto Sparkle.framework failed: {result.stderr.decode()}')
    else:
        print(f'\nWARNING: vendored Sparkle.framework not found at {sparkle_src}')

    # Step 2: inject Sparkle keys + version metadata into Info.plist via
    # PlistBuddy. Sparkle refuses to start (error code 7) if CFBundleVersion
    # is missing or CFBundleShortVersionString is the PyInstaller default
    # "0.0.0" — both must reflect the real app version. Same value works
    # for both since we don't maintain a separate build number.
    info_plist = os.path.join(dist_root, 'Contents', 'Info.plist')
    pb = '/usr/libexec/PlistBuddy'
    version_file = os.path.join(os.path.dirname(__file__), 'version.txt')
    app_version = '0.0.0'
    if os.path.isfile(version_file):
        with open(version_file) as vf:
            app_version = vf.read().strip() or '0.0.0'
    if os.path.isfile(info_plist) and sparkle_pubkey:
        sparkle_keys = [
            ('SUFeedURL', 'string', appcast_url),
            ('SUPublicEDKey', 'string', sparkle_pubkey),
            ('SUEnableAutomaticChecks', 'bool', 'true'),
            ('SUScheduledCheckInterval', 'integer', '86400'),
            ('CFBundleVersion', 'string', app_version),
            ('CFBundleShortVersionString', 'string', app_version),
        ]
        print('\nInjecting Sparkle keys into Info.plist:')
        for key, ktype, value in sparkle_keys:
            r = subprocess.run([pb, '-c', f'Set :{key} {value}', info_plist],
                               capture_output=True)
            if r.returncode != 0:
                r = subprocess.run([pb, '-c', f'Add :{key} {ktype} {value}', info_plist],
                                   capture_output=True)
            if r.returncode == 0:
                print(f'  {key} = {value}')
            else:
                print(f'  WARNING: failed to set {key}: {r.stderr.decode().strip()}')
    elif not sparkle_pubkey:
        print('\nWARNING: no Sparkle public key — skipping Info.plist injection')

    # Step 3: re-sign the outer .app with an ad-hoc signature. Both copying
    # Sparkle.framework into Contents/Frameworks and patching Info.plist
    # invalidate PyInstaller's seal on the outer bundle. Sparkle 2.x's update
    # validator rejects updates whose outer seal is broken (the inner
    # frameworks remain validly signed, so we only re-seal the outermost
    # bundle). errSecCSBadResource (-67030) is the symptom; this is the cure.
    # Confirmed by Sparkle maintainer that ad-hoc + EdDSA is supported.
    print('\nRe-signing outer .app bundle (ad-hoc) to repair Info.plist seal...')
    cs = subprocess.run(
        ['codesign', '--force', '--sign', '-', dist_root],
        capture_output=True,
    )
    if cs.returncode != 0:
        print(f'  WARNING: codesign failed: {cs.stderr.decode().strip()}')
    else:
        verify = subprocess.run(
            ['codesign', '--verify', '--verbose=2', dist_root],
            capture_output=True,
        )
        if verify.returncode == 0:
            print('  outer bundle signature: valid')
        else:
            print(f'  WARNING: outer bundle still invalid: {verify.stderr.decode().strip()}')

if sys.platform == 'win32':
    # Place WinSparkle.dll at the top of the bundle (next to the .exe) so
    # Windows' default DLL search finds it without PATH manipulation.
    winsparkle_src = os.path.join(os.path.dirname(__file__), 'vendored', 'winsparkle', 'WinSparkle.dll')
    winsparkle_dest = os.path.join(dist_root, 'WinSparkle.dll')
    if os.path.isfile(winsparkle_src):
        shutil.copy2(winsparkle_src, winsparkle_dest)
        sz = os.path.getsize(winsparkle_dest) / 1024 / 1024
        print(f'\nWinSparkle.dll copied into bundle ({sz:.1f} MB)')
    else:
        print(f'\nWARNING: vendored WinSparkle.dll not found at {winsparkle_src}')

    # Write bundled torch + torchvision versions into _internal so the GPU override
    # runtime hook and in-app installer can match the correct CUDA wheels.
    try:
        import torch as _torch
        _torch_ver = _torch.__version__
        _ver_dest = os.path.join(dist_root, '_internal', 'stc_expected_torch_version.txt')
        with open(_ver_dest, 'w') as _f:
            _f.write(_torch_ver)
        print(f'\nWrote stc_expected_torch_version.txt: {_torch_ver}')
    except Exception as _e:
        print(f'\nWARNING: could not write stc_expected_torch_version.txt: {_e}')
    try:
        import torchvision as _tv
        _tv_ver = _tv.__version__
        _tv_dest = os.path.join(dist_root, '_internal', 'stc_expected_torchvision_version.txt')
        with open(_tv_dest, 'w') as _f:
            _f.write(_tv_ver)
        print(f'Wrote stc_expected_torchvision_version.txt: {_tv_ver}')
    except Exception as _e:
        print(f'WARNING: could not write stc_expected_torchvision_version.txt: {_e}')

after = dir_size_mb(os.path.join('dist'))
print(f'\nAfter cleanup: {after:.1f} MB  (saved {before - after:.1f} MB)')

sys.exit(0)
