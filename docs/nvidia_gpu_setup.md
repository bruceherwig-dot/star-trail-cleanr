# NVIDIA GPU Setup for Star Trail CleanR (Windows)

**Status: experimental.** This procedure has not yet been verified on a real Windows + NVIDIA machine. You are the first guinea pig. The "Roll back" section at the bottom puts your install back to CPU mode safely if anything breaks. Please email me how it went either way.

By default Star Trail CleanR ships with the CPU-only build of PyTorch to keep the installer small. This guide replaces it with the NVIDIA / CUDA build so the trail detector runs on your GPU instead of your CPU. Expected speedup is roughly 5-10x on modern NVIDIA cards.

These instructions are for Star Trail CleanR **v2.04-beta** (which uses PyTorch 2.11.0 and torchvision 0.26.0 internally). If you are on a different version, the file paths and the exact wheels to download may differ — email me first.

## What you need

- **NVIDIA GPU.** Any RTX card, or GTX 10-series and newer.
- **NVIDIA driver** version 525 or newer. Run `nvidia-smi` from a Command Prompt to check; if the command isn't found, install the latest "Game Ready" driver from nvidia.com first.
- **Windows 10 or 11**, 64-bit.
- **Star Trail CleanR v2.04-beta installed.** Note where you installed it. The installer's default is `C:\Program Files\Star Trail CleanR\` but the user-mode default is `%LOCALAPPDATA%\Programs\StarTrailCleanR\`. The instructions below say "your install folder" — replace with whichever you used.
- **Admin rights** if you installed into Program Files. Not needed if installed into LocalAppData.
- **About 4 GB free disk space.**
- **A tool that can extract `.whl` files.** A `.whl` (Python wheel) is just a renamed zip. 7-Zip or WinRAR work fine.

## Step 1: Download the two CUDA wheels

Click each link to download. Both are official PyTorch builds from pytorch.org.

- **torch 2.11.0 + CUDA 12.8** (~2.6 GB):
  https://download.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp311-cp311-win_amd64.whl

- **torchvision 0.26.0 + CUDA 12.8** (~9 MB):
  https://download.pytorch.org/whl/cu128/torchvision-0.26.0%2Bcu128-cp311-cp311-win_amd64.whl

(If your driver is too old for CUDA 12.8, swap `cu128` for `cu126` in both URLs to use CUDA 12.6 instead.)

## Step 2: Quit Star Trail CleanR

Make sure the app is fully closed. Open Task Manager (Ctrl+Shift+Esc), look for any `StarTrailCleanR.exe` or related processes, end them. Files inside an open app cannot be replaced.

## Step 3: Extract the wheels

Right-click each `.whl` file and "Extract" with 7-Zip or your tool of choice. (If your tool refuses to open `.whl`, rename the file extension to `.zip` first.)

After extracting `torch-2.11.0+cu128-cp311-cp311-win_amd64.whl` you'll get a folder structure like:
```
torch/                       (this is one of the folders you need)
torch-2.11.0+cu128.dist-info/    (the matching dist-info)
... possibly other folders like torchgen/, functorch/ — copy those too if present
```

After extracting torchvision's wheel:
```
torchvision/
torchvision-0.26.0+cu128.dist-info/
```

## Step 4: Back up the existing CPU folders

In your install folder's `_internal` directory (e.g., `C:\Program Files\Star Trail CleanR\_internal\`), **rename** these four items rather than deleting them:

- `torch` → `torch.cpu.bak`
- `torch-2.11.0.dist-info` → `torch-2.11.0.dist-info.cpu.bak`
- `torchvision` → `torchvision.cpu.bak`
- `torchvision-0.26.0.dist-info` → `torchvision-0.26.0.dist-info.cpu.bak`

Renaming instead of deleting means you can roll back instantly if anything goes wrong.

## Step 5: Copy the CUDA folders in

From the extracted wheels (Step 3), copy the folders into `_internal`:

- `torch/` → `_internal\torch\`
- `torch-2.11.0+cu128.dist-info/` → `_internal\torch-2.11.0+cu128.dist-info\`
- `torchvision/` → `_internal\torchvision\`
- `torchvision-0.26.0+cu128.dist-info/` → `_internal\torchvision-0.26.0+cu128.dist-info\`

If Windows asks for admin rights, allow.

## Step 6: Launch Star Trail CleanR

Open the app the normal way. If the swap worked, your runs should be noticeably faster. There is no GUI indicator yet that says "GPU active" (it's on the to-do list). For now, the test is:

- Run a small batch (say, 20 frames) before the swap and note the time.
- Run the same batch after the swap. If it's 5-10x faster, the GPU is working.

You can also open Task Manager → Performance → GPU during a run and watch for activity on your NVIDIA card.

## If something goes wrong — roll back to CPU

If the app crashes on launch, refuses to detect frames, or behaves weirdly:

1. Quit Star Trail CleanR.
2. In `_internal`, **delete** the four CUDA folders you just copied in.
3. **Rename** the four `.cpu.bak` folders back to their original names (drop the `.cpu.bak` suffix).
4. Launch the app. You're back to the CPU build, no harm done.

## Please report

Whether it worked or not, drop me a note at **bruceherwig@gmail.com** with:

- Your GPU model
- Your driver version (`nvidia-smi` output is perfect)
- Whether the swap worked
- Rough before/after speed if it did
- Any error messages or weird behavior

Each report helps refine these instructions. If a few people get this working, GPU support will become the default install path for NVIDIA users in a future release.
