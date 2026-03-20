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

# Always include the algorithm script
add_data = [f'astro_clean_v4.py{sep}.']
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
    '--onedir', '--windowed', 'star_trail_cleanr.py',
    '--name', 'StarTrailCleanR',
]
for d in add_data:
    cmd += ['--add-data', d]

print(f'Bundling data from {len(seen)} packages:')
for pkg in sorted(seen):
    print(f'  {pkg}')

result = subprocess.run(cmd)
sys.exit(result.returncode)
