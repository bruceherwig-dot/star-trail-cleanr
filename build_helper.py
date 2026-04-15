#!/usr/bin/env python3
"""
Auto-discovers all installed packages that contain data files and runs PyInstaller.
Prevents missing-data-file crashes without requiring a manually maintained list.
"""
import os, site, subprocess, sys

sep = ';' if sys.platform == 'win32' else ':'

# Extensions PyInstaller handles natively — exclude from --add-data
SKIP_EXT = {'.py', '.pyc', '.pyo', '.pyd', '.so', '.dylib', '.dll'}

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
            f'modules{sep}modules']
seen = set()

for site_dir in site_dirs:
    if not os.path.isdir(site_dir):
        continue
    for pkg_name in sorted(os.listdir(site_dir)):
        if pkg_name in seen:
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

cmd = [
    sys.executable, '-m', 'PyInstaller',
    '--onedir', '--windowed', '--noupx', 'star_trail_cleanr.py',
    '--name', 'StarTrailCleanR',
    '--collect-all', 'cv2',
    '--collect-all', 'numpy',
    '--collect-all', 'PySide6',
    '--collect-all', 'sahi',
    '--collect-all', 'ultralytics',
]
for d in add_data:
    cmd += ['--add-data', d]

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
