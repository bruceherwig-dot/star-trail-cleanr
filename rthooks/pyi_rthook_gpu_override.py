import os
import sys

if sys.platform != 'win32' or not hasattr(sys, '_MEIPASS'):
    pass
else:
    _localappdata = os.environ.get('LOCALAPPDATA', '')
    _override_dir = os.path.join(_localappdata, 'StarTrailCleanR', 'gpu_override')
    _ver_file = os.path.join(_override_dir, 'torch_version.txt')

    if os.path.isdir(_override_dir) and os.path.isfile(_ver_file):
        try:
            with open(_ver_file) as _f:
                _override_ver = _f.read().strip().split('+')[0]

            _expected_ver = None
            _expected_file = os.path.join(sys._MEIPASS, 'stc_expected_torch_version.txt')
            if os.path.isfile(_expected_file):
                with open(_expected_file) as _f:
                    _expected_ver = _f.read().strip().split('+')[0]

            if _expected_ver and _override_ver == _expected_ver:
                sys.path.insert(0, _override_dir)
            else:
                os.environ['STC_GPU_VERSION_MISMATCH'] = '1'
        except Exception:
            pass
