# NVIDIA GPU Setup for Star Trail CleanR (Windows)

**Status: experimental.** This procedure has not yet been verified on a real Windows + NVIDIA machine. You are the first guinea pig. The "Roll back" section at the bottom puts your install back to CPU mode safely if anything breaks. Please email me how it went either way.

By default Star Trail CleanR ships with the CPU-only build of PyTorch to keep the installer small. This guide sets up a permanent GPU pack in a folder that Star Trail CleanR updates never touch. You do this once, and every future update continues to use your GPU automatically.

## What you need

- **NVIDIA GPU.** Any RTX card, or GTX 10-series and newer.
- **NVIDIA driver** version 525 or newer. Run `nvidia-smi` from a Command Prompt to check; if the command isn't found, install the latest "Game Ready" driver from nvidia.com first.
- **Windows 10 or 11**, 64-bit.
- **Star Trail CleanR installed.**
- **About 4 GB free disk space.**
- **A tool that can extract `.whl` files.** A `.whl` (Python wheel) is just a renamed zip. 7-Zip or WinRAR work fine.

## Step 1: Find out which PyTorch version this release expects

Open your Star Trail CleanR install folder in Windows Explorer. The default locations are:
- `C:\Program Files\StarTrailCleanR\` (if you installed for all users)
- `%LOCALAPPDATA%\Programs\StarTrailCleanR\` (user-only install, the default)

Inside the install folder, open `_internal\stc_expected_torch_version.txt`. It contains a single line like `2.8.0`. That number is the PyTorch version you need to match when downloading the CUDA wheel in Step 2.

## Step 2: Download the two CUDA wheels

Use the version number from Step 1 to build the download URLs. Replace `X.Y.Z` with your version number and pick a CUDA suffix based on your driver:

- CUDA 12.8 (driver 520+, recommended): use `cu128`
- CUDA 12.6 (driver 520+): use `cu126`
- CUDA 11.8 (older cards, driver 450+): use `cu118`

Download both wheels from pytorch.org (replace `X.Y.Z` and `cuXXX` with your values):

- **torch**: `https://download.pytorch.org/whl/cuXXX/torch-X.Y.Z%2BcuXXX-cp311-cp311-win_amd64.whl`
- **torchvision**: `https://download.pytorch.org/whl/cuXXX/torchvision-0.Y.Z%2BcuXXX-cp311-cp311-win_amd64.whl`

(torchvision's minor version tracks torch's — if torch is 2.8.0, torchvision is 0.23.0. The PyTorch website's "install" page lists the exact matching pair for any given release.)

## Step 3: Quit Star Trail CleanR

Make sure the app is fully closed before continuing.

## Step 4: Create the GPU pack folder

Open Windows Explorer and navigate to:

```
%LOCALAPPDATA%\StarTrailCleanR\
```

Create a new folder called `gpu_override` inside it. The full path should be:

```
%LOCALAPPDATA%\StarTrailCleanR\gpu_override\
```

Star Trail CleanR updates never write to this location, so your GPU files are safe across every future update.

## Step 5: Extract the wheels into the GPU pack folder

Right-click each `.whl` file and extract with 7-Zip or your tool of choice. (If your tool refuses, rename the `.whl` extension to `.zip` first.)

From the extracted `torch` wheel, copy these folders into `gpu_override\`:
- `torch\`
- `torch-X.Y.Z+cuXXX.dist-info\`
- Any other top-level folders present (e.g., `torchgen\`, `functorch\`)

From the extracted `torchvision` wheel, copy:
- `torchvision\`
- `torchvision-0.Y.Z+cuXXX.dist-info\`

## Step 6: Create the version marker file

In the `gpu_override\` folder, create a plain text file named exactly:

```
torch_version.txt
```

Open it in Notepad and type the full version string of the torch wheel you downloaded — including the CUDA suffix. For example:

```
2.8.0+cu128
```

Save and close. This file is how Star Trail CleanR knows your GPU pack matches this release. If you update to a new Star Trail CleanR version that uses a different PyTorch version, the app will detect the mismatch, fall back to CPU, and show a message in Settings telling you to reinstall the GPU pack.

## Step 7: Launch Star Trail CleanR

Open the app the normal way. If the swap worked, your runs should be noticeably faster. Open the **Settings** tab and look at the **Compute Device** section — it should read "NVIDIA CUDA — GPU acceleration active."

You can also open Task Manager → Performance → GPU during a run and watch for activity on your NVIDIA card.

## When Star Trail CleanR updates

Nothing to do. The update installs into the app folder; it never touches `%LOCALAPPDATA%\StarTrailCleanR\gpu_override\`. Your GPU pack loads automatically after every update, as long as the PyTorch version hasn't changed.

If the PyTorch version does change in a new release, the Settings tab will show: "GPU pack version mismatch — reinstall the GPU pack for this version." Repeat Steps 1–6 with the new version number to get back on GPU.

## Roll back to CPU

If the app crashes on launch, refuses to detect frames, or behaves weirdly:

1. Quit Star Trail CleanR.
2. Open `%LOCALAPPDATA%\StarTrailCleanR\`.
3. Rename `gpu_override` to `gpu_override.bak` (or delete it entirely).
4. Launch the app. You're back to the CPU build, no harm done.

## Please report

Whether it worked or not, drop me a note at **bruceherwig@gmail.com** with:

- Your GPU model
- Your driver version (`nvidia-smi` output is perfect)
- Whether the swap worked
- Rough before/after speed if it did
- Any error messages or weird behavior

Each report helps refine these instructions. If a few people get this working, GPU support will become the default install path for NVIDIA users in a future release.
