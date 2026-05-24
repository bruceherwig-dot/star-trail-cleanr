# NVIDIA GPU Setup for Star Trail CleanR (Windows)

GPU acceleration makes Star Trail CleanR noticeably faster on NVIDIA machines. On most systems, one click in Settings handles the entire installation. This page covers what to do when that fails.

---

## Normal installation

1. Open Star Trail CleanR.
2. Go to **Settings**.
3. Click **Install GPU Support**.
4. Wait for the download (approximately 3-4 GB). Depending on your connection, this takes 5-15 minutes.
5. Click **Restart Now** when prompted.

After restart, the Settings page will confirm GPU acceleration is active.

---

## If you see "Download blocked (HTTP 403)"

PyTorch's download servers restrict access from some networks and regions. This is not a Star Trail CleanR problem. There are two ways to work around it.

### Option 1: Use a VPN (recommended, fastest)

A VPN routes the download through a server in an unrestricted region.

1. Install a VPN. Free options that work well: [ProtonVPN](https://protonvpn.com) (free tier) or [Windscribe](https://windscribe.com) (free tier). Any paid VPN you already have works too.
2. Connect to any US or European server.
3. Open Star Trail CleanR and go to **Settings**.
4. Click **Install GPU Support**.
5. Let the download complete, then click **Restart Now**.
6. Once the app restarts and GPU is active, you can disconnect the VPN. It is not needed again.

### Option 2: Download from an alternative server (no VPN needed)

Alibaba Cloud hosts a complete copy of the PyTorch wheel library, including the same Windows CUDA files. It is not regionally restricted. You can download the wheel files manually from there and install them using the manual steps below.

Use this base URL in place of the pytorch.org URL when building the download links in Step 3 of the manual installation section:

```
https://mirrors.aliyun.com/pytorch-wheels/cu128/
```

---

## Manual installation

Use this method when automatic installation fails for any reason.

### What you need

- NVIDIA GPU: any RTX card, or GTX 10-series or newer
- NVIDIA driver version 525 or newer (run `nvidia-smi` in Command Prompt to check; if the command is not found, install the latest Game Ready driver from nvidia.com first)
- Windows 10 or 11, 64-bit
- About 4 GB free disk space
- 7-Zip or WinRAR to extract the wheel files

### Step 1: Find the expected PyTorch version

Open your Star Trail CleanR install folder in File Explorer. Default locations:

- `C:\Program Files\StarTrailCleanR\` (all-users install)
- `%LOCALAPPDATA%\Programs\StarTrailCleanR\` (user-only install, the default)

Inside the install folder, open `_internal\stc_expected_torch_version.txt`. It contains one line like `2.8.0`. This is the version number you need in the next step.

### Step 2: Find the matching torchvision version

Go to [pytorch.org/get-started/previous-versions](https://pytorch.org/get-started/previous-versions/) and find the entry matching your torch version. The torchvision version is listed alongside it. For example: torch 2.8.0 pairs with torchvision 0.23.0.

### Step 3: Download the two wheel files

Build the download URLs by replacing `TORCH_VER` and `TV_VER` with your version numbers.

If pytorch.org is blocked, replace `https://download.pytorch.org/whl/cu128` with `https://mirrors.aliyun.com/pytorch-wheels/cu128` in both URLs below. The file paths are identical on both servers.

**torch** (approximately 2.5 GB):
```
https://download.pytorch.org/whl/cu128/torch-TORCH_VER%2Bcu128-cp311-cp311-win_amd64.whl
```

**torchvision** (approximately 1 GB):
```
https://download.pytorch.org/whl/cu128/torchvision-TV_VER%2Bcu128-cp311-cp311-win_amd64.whl
```

### Step 4: Close Star Trail CleanR

Make sure the app is fully closed before continuing.

### Step 5: Create the GPU pack folder

In File Explorer, navigate to:
```
%LOCALAPPDATA%\StarTrailCleanR\
```

Create a new folder named `gpu_override`. The full path should be:
```
%LOCALAPPDATA%\StarTrailCleanR\gpu_override\
```

Star Trail CleanR updates never write to this location. Your GPU files survive every future app update automatically.

### Step 6: Extract the wheels into the GPU pack folder

Right-click each `.whl` file and open with 7-Zip or WinRAR. If your tool refuses, rename the `.whl` extension to `.zip` first.

From the torch wheel, copy into `gpu_override\`:
- `torch\`
- `torch-TORCH_VER+cu128.dist-info\`
- Any other top-level folders present (such as `torchgen\` or `functorch\`)

From the torchvision wheel, copy into `gpu_override\`:
- `torchvision\`
- `torchvision-TV_VER+cu128.dist-info\`

### Step 7: Create the version marker file

In `gpu_override\`, create a plain text file named exactly:
```
torch_version.txt
```

Open it in Notepad and type the torch version string including the CUDA suffix:
```
TORCH_VER+cu128
```

For example: `2.8.0+cu128`. Save and close. This file tells Star Trail CleanR that your GPU pack matches the current app version.

### Step 8: Launch Star Trail CleanR

Open the app the normal way. If the installation worked, Settings will confirm GPU acceleration is active. You can also open Task Manager, go to Performance, select your NVIDIA GPU, and watch for activity during a run.

---

## After a Star Trail CleanR update

Updates install into the app folder only. The `gpu_override` folder is never modified. GPU acceleration continues to work automatically after every update.

If the app updates to a new PyTorch version, Settings will show a version mismatch warning. Repeat Steps 1-8 with the new version number to get back on GPU.

---

## Roll back to CPU

If the app crashes or behaves incorrectly after installation:

1. Quit Star Trail CleanR.
2. Open `%LOCALAPPDATA%\StarTrailCleanR\` in File Explorer.
3. Rename `gpu_override` to `gpu_override_bak` (or delete it entirely).
4. Launch the app. It returns to CPU mode immediately.

You can also roll back from inside the app: go to **Settings** and click **Clear GPU Support Files**.

---

## Questions or problems

Email **bruceherwig@gmail.com** with your GPU model, NVIDIA driver version, and a description of what happened. Reports help improve the installer for everyone.
