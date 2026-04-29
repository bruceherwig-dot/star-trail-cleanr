# Star Trail CleanR, Version History

---

## v1.93-beta
- **Fix: Apple Silicon Mac users no longer crash during the AI warmup step.** The version of PyTorch we ship hadn't implemented one of the operations (NMS) for Apple's GPU yet, so the model crashed the moment it tried to run on the GPU — which is the default on every M-series Mac. The fix tells PyTorch to fall back to the CPU for that one operation, invisible to the user, no real performance impact. Crash reports came in from two Apple Silicon testers within 24 hours, which is how we caught it.
- **Bundled font for consistent appearance across Mac, Windows, and Linux.** Earlier versions relied on each operating system's default font (San Francisco on Mac, Segoe UI on Windows). The same point size renders at different widths in those fonts, which caused some controls to clip text on Windows even though they looked fine on Mac (most visibly the JPEG quality field). The app now ships with the open-source Inter font and forces every label and button to render in it, so widget widths are identical on every platform.
- **Setup page tightened so all six steps and the Clean My Stars button fit on first launch.** Hint text now sits next to each step heading instead of on its own line, vertical spacing reduced, and the window opens at 1100x950 centered on the screen the first time you run it. Below that size the action button always stays visible.
- **The "run complete" summary is now a centered popup window.** It used to be an inline panel that fought the log area for vertical space and could partially cover the Back to Setup button. Now the popup opens centered, shows the full summary (trails swept, time saved, estimate vs actual), and has Open Cleaned Folder and Close buttons. Closing leaves you on a clean processing page with the full log visible.
- **Step 1 reworded to "Select Folder with Your Star Trail Images".** Some Windows testers thought they should pick individual image files because Windows' folder picker doesn't show file thumbnails the way Mac's does. The new wording makes the requested action unambiguous.
- **Many cross-platform polish fixes.** Close button "X" now centers cleanly. Support button heart is the right size again. JPEG quality field width fits on every OS. FAQ and About text scaled correctly for the new font. Click thresholds tuned for Mac trackpads (a soft tap now registers as a click everywhere it should).

## v1.92-beta
- **Fix: app no longer crashes when loading 16-bit TIFFs on Windows.** A Windows tester running 16-bit TIFFs through v1.91 hit "Cannot handle this data type: (1, 1, 3), <u2" the moment a batch started. Root cause: the trail detector was re-reading the file from disk with an OpenCV call that's supposed to convert 16-bit images to 8-bit, but on Windows for certain Lightroom-exported TIFFs that conversion silently doesn't happen, and a 16-bit array slipped into a library downstream that has no 16-bit color mode. Two-part fix: the worker now hands the detector its already-prepared 8-bit copy directly (no redundant disk read), and the detector itself has a defensive normalizer that forces every input to 8-bit, 3-channel color before the AI sees it. Whatever bizarre TIFF a user feeds in, it cannot crash the detector again. Output 16-bit precision is unchanged: the 8-bit copy is just for the AI's eyes; original 16-bit pixels still flow through repair and out to a real 16-bit TIFF if you picked that format. Also rewrote the regression test using a real-world 16-bit TIFF writer so this exact failure can never silently ship again.
- **Crash reports from inside the cleaning step now reach the developer.** Previous versions only reported crashes that happened in the main app window. The actual cleaning work runs in a separate background process, and crashes there were invisible. Two new safety nets: the background process now reports its own crashes when the user has opted in, and the main app captures any background-process error text and forwards it as well, so even crashes that die before reporting can start are still caught. Privacy is unchanged: nothing is sent if the user did not opt in, and no images, paths, or personal information are collected.

## v1.91-beta
- **Fix: TIFF 16-bit output no longer crashes.** A Windows tester saw "Cannot handle this data type: (1, 1, 3), <u2" when he picked TIFF 16-bit as his output format. Turns out the line of code that writes 16-bit TIFFs has been broken since v1.0-beta (it asked Pillow to do something Pillow does not actually support), but you could only reach it once you had a 16-bit TIFF input working — which v1.9-beta is the first version to support. So the moment 16-bit input was unblocked, the latent output bug surfaced. The 16-bit TIFF write now uses a different library (tifffile) that handles 16-bit RGB cleanly. Pixel values, color profile, and DPI all preserved through the write. Most users will never have noticed because the default output format is JPG.

## v1.9-beta
- **Fix: app no longer crashes on first run.** v1.81-beta would fail on the very first batch with a "ModuleNotFoundError: No module named 'skimage'" message. The error was triggered by an unused import that fired before the first frame even loaded. Removed.
- **Fix: log lines now read correctly on Windows.** Some users saw garbled characters like "_DSC0023.tif â€" 0 trails" in the processing log. The worker was writing UTF-8 text but Windows was reading it as a different encoding by default. Fixed on both ends: the worker now uses plain ASCII in log lines, and the reader is forced to UTF-8.
- **Setup page now scrolls.** On smaller laptop screens, especially Windows laptops at 100% DPI, some users saw the section headings clipped at the top with no way to reach Step 6 or the Clean My Stars! button. The setup page now scrolls vertically when the window isn't tall enough to show everything at once.
- **Run summary saved to disk.** At the end of every cleaning run, the app now writes a small text file alongside the cleaned images with the run details: how many frames, how many trails removed, elapsed time, estimate vs. actual. Useful for sharing results or comparing runs.
- **First Linux release.** A Linux 64-bit build is now available alongside Mac (Apple Silicon and Intel) and Windows. Tested on Ubuntu 22.04 and newer, Debian 12 and newer, Fedora 36 and newer. Download the tar.gz, extract, run the StarTrailCleanR binary inside.

## v1.81-beta
- **16-bit TIFF input now works on Windows.** A Windows tester reported a crash on a 50-frame run of 16-bit TIFFs exported from Lightroom (Nikon Z6ii). The trail detector was handing the file path to its scanning library, which then re-opened the file with a loader that doesn't understand 16-bit color and crashed with a cryptic data-type error. Fix: hand the scanning library the already-loaded image directly, bypassing its built-in loader entirely. 16-bit TIFFs from any camera and any export tool now run cleanly on both Mac and Windows. Also added a regression test so this exact failure can never silently come back.
- **Now ships with Trail Detector v3 out of the box.** Earlier versions bundled Trail Detector v2 and offered v3 as an in-app download via the orange update banner. If the banner didn't reach a user (network blocked, dismissed, or the user started a run before the background check finished), they were stuck on v2. v1.81-beta bundles v3 directly so every new install starts on the latest detector. Existing users who already downloaded v3 through the banner are unaffected.
- **Slope-match merge from v1.8-beta has been turned off.** v1.8-beta added a step that tried to merge trail detections that crossed between the AI's scanning windows. Field testing on real frames showed the merge was producing visible artifacts on some trails (oversized repair zones, neighboring trails being merged when they shouldn't be). The merge is now off; the app falls back to the cleaner v1.7-beta detection behavior. We'll revisit cross-window stitching with a different approach in a future release.

## v1.8-beta
- **Trails that span more than one of the AI's scanning windows are now stitched back together.** Long satellite trails that cross multiple tiles used to come out as several disconnected pieces in the cleaned output, because the existing duplicate-remover step looked at bounding-box overlap and didn't know trails are long and thin. A new merging step glues those pieces back into one trail when they really are pieces of the same physical streak — same slope, sitting on the same line, with masks that actually share pixels. Cleaner repair zones on long trails, less stair-stepping at tile boundaries.

## v1.73-beta
- **Optional anonymous crash reporting.** The first time you launch this version, the app asks if you'd like to send anonymous crash reports. If you say yes, the app sends an automatic report (stack trace, operating system, app version) when something crashes, so the bug can be found and fixed. If you say no, nothing is sent. Either way, no images, no folder paths, and no personal information are ever collected. Helps the developer fix problems users hit in the wild without making them email a bug report.

## v1.72-beta
- **Light + dark mode now both render correctly.** Every banner, button, card, and tab has been wired through one central color list with light and dark variants. Section headings, hint text, and disabled buttons all read properly in both modes. If you toggle macOS Light/Dark while the app is open, it relaunches automatically with your folder selections preserved.
- **FAQ and About tabs have breathing room.** The text inside each tab no longer hugs the edges of the panel.
- **Desktop launcher cleanup.** Only one Star Trail CleanR icon shows in the dock now instead of two. The launcher quits itself the moment it has handed off to the running app. Developer-only change; doesn't affect end users.

## v1.71-beta
- **Cleaned files now carry a Star Trail CleanR stamp.** Open any cleaned image in Photoshop, Lightroom, macOS Finder, or Windows Explorer and the Description / Software / Comments field reads "Star Trail CleanR v1.71 / Trail Detector v3 / www.startrailcleanr.com". All original camera info (make, model, lens, exposure, date) is preserved unchanged.
- **DPI metadata preserved.** If your source images are 300 DPI, the cleaned output stays 300 DPI instead of being reset to 72. Same idea for any other DPI value. Purely cosmetic fix but avoids confusion in print workflows.
- **Mac app icon renders at the right size in the Dock.** The app icon now follows Apple's Big Sur safe-area spec (824 pixel design inside a 1024 pixel frame with transparent margin). The previous icon filled the full frame, so macOS rendered it larger than every other Mac app icon. No change on Windows, which renders full-bleed by design.
- **Desktop dev-mode icon.** When Bruce launches the app from his Desktop AppleScript wrapper, the running process's Dock icon now shows the Star Trail CleanR icon instead of the generic Python rocket. Developer-only change; doesn't affect end users.

## v1.7-beta
- **Tighter trail repair.** The app was occasionally painting over more sky than it needed around a trail. When a trail sat right on the border between two of the AI's scanning windows, both windows detected the same trail and the step that combined them unioned the two detections into one inflated shape. Fixed by keeping the higher-confidence detection and dropping the duplicate, instead of merging them. Cleaner repair zones, especially noticeable around bright stars sitting close to a trail.

## v1.6-beta
- **Open Folder buttons** next to Browse in Steps 1 and 2 of the setup page. Click to jump straight to that folder in Finder (Mac) or Explorer (Windows). Greyed out when the path field is empty or the folder doesn't exist yet.
- **Image count setting is now sticky**: changing the input folder no longer resets your "Number of Images to Process" choice. It stays wherever you left it.
- **Trail Detector version now shows a "v" prefix** in the header (e.g., "Trail Detector v3") to match how releases are tagged on GitHub.
- **NVIDIA GPU detection**: if you have an NVIDIA graphics card, the app now detects it at launch and shows a small banner letting you know full GPU support is coming in a future update. No action needed; dismissible.

## v1.5-beta
- **Update check on startup**: when a newer version of Star Trail CleanR is released, the app now shows a banner with a Download button. Clicking opens your browser to the download page. Nothing auto-installs; you stay in control.
- **Trail detector updates**: when a new trail detector is released, the app shows a card with the name, what's better, and credits to community contributors. Click Download to pull the new detector; it takes over on your next run. Click Not right now to skip for this launch.
- **Active detector shown in the header**: under the version number, the header now shows which Trail Detector is currently loaded (e.g., "Trail Detector 2").
- **New Mac installer for Intel users**: the download page now offers a separate "Mac (Intel)" installer for older Intel-based Macs. Apple Silicon users keep using the existing Mac Apple Silicon download.
- **Automatic hardware selection**: the app now picks the best available hardware at runtime. NVIDIA first if you have one, then Apple's fast-processing mode on Apple Silicon, then regular CPU. No setup needed.

## v1.4-beta
- Windows installer now ships inside a zip wrapper. Microsoft Edge was quarantining the unsigned installer with a Defender SmartScreen warning whose "Keep" option was buried in a hidden dropdown next to the "Delete" button, and most novice users never found it. Wrapping the installer in a zip sidesteps that gate entirely. Download the zip, right-click and choose "Extract All...", then double-click StarTrailCleanRSetup.exe inside the extracted folder. The familiar "Windows protected your PC" warning still appears at install time and is handled the same way ("More info" then "Run anyway"). Mac unchanged.

## v1.3-beta
- Stable download links: the Mac zip and Windows installer can now be linked from one permanent URL each, no more updating links every release.
- JPEG quality default raised from 80 to 95 to eliminate visible 8x8 block artifacts that showed up in the sky after stacking 100+ frames. Old default was inherited from web-image conventions and was wrong for star-trail stacking.
- End-of-run summary now reads "airplane and satellite trails" instead of just "airplane trails", which matches what the app actually removes.
- Resolution check is faster and quieter: no more "scanning 1/22, 2/22..." lines on every batch. The check happens once at the start of the run; batches just load straight into processing.
- "Loading YOLO model..." renamed to "Loading AI trail detector..." in the run log.

## v1.2-beta
- Windows now ships as a one-click installer instead of a raw zip. The new Setup file is a single .exe that installs the app to Program Files, creates a Start Menu shortcut, and registers a real uninstaller. No more 60,000-file Explorer extract.
- The installer is much smaller than the old zip thanks to LZMA2 compression.
- Mac unchanged for now. Mac handles the .app-in-zip cleanly because macOS treats the bundle as a single item.

## v1.1-beta
- Fixed a crash on Windows where the app tried to load the AI model from a local folder that doesn't exist on the tester's machine. The model is now bundled inside the app itself.

## v1.0-beta
A full rewrite. Everything below is new since v0.19-beta.

**Release-testing fixes (v1.006–v1.009):**
- Fixed a crash on first cleaning run caused by a missing math library in the frozen app
- Mask Painter: Back button now looks and behaves like a real button
- Mask Painter: cursor switches back to a normal arrow when you move off the image into the gray margin
- Mask Painter: zoom now anchors to the center of the view instead of the top
- Mask Painter: brush-size scroll step smoothed out so the brush grows at a sensible pace on trackpads and mice

**Main v1.0-beta features:**
- Native desktop app (macOS and Windows), no more browser window
- New AI trail detector trained on thousands of real astrophotography frames, including a community dataset from gkyle
- Star Bridge repair that borrows clean pixels from the frames before and after, so removed trails blend in seamlessly
- Silent ground-only hot-pixel fix, cleans stuck and dead pixels on the landscape without touching the sky
- Dark mode and a proper tabbed interface
- Logo and banner across the top
- Mask Painter tool for editing the ground mask when you want manual control
- "Scrubbing the stars" run screen with live progress, time elapsed, trails swept, and an estimated time to finish
- End-of-run stats: total trails removed, total frames cleaned, total time, and an estimate of how much manual editing you just skipped
- JPG or TIF output with a JPEG quality slider
- Live frame count when you pick a folder
- All labels are selectable so you can copy paths, values, and numbers
- Have a suggestion? There's a mailto link in the About tab

## v0.19-beta
- Fixed "not open anymore" error on Mac when relaunching after closing the browser tab
- Improved time estimate accuracy for high-resolution cameras — a 36MP camera now shows a realistic estimate instead of the 20MP reference time
- Updated accepted file types label in the interface (.JPG, .TIF 8 & 16 bit)

## v0.18-beta
- Fixed batch count estimate — the "Est. batches" number now always matches the actual number of batches processed

## v0.17-beta
- Fixed relaunch on Mac: closing the browser tab and reopening the app now brings back the existing session instead of showing "application not open" error

## v0.16-beta
- Added 16-bit TIFF support — the app now correctly detects and processes 16-bit TIF files without removing stars
- Improved detection accuracy for high-resolution cameras (35MP, 45MP, etc.) — processing time and false detection counts now scale correctly with image size

## v0.15-beta
- Fixed Unicode crash on Windows when processing files with special characters in output
- Removed hardcoded Mac file paths that caused errors on other systems
- Improved placeholder text throughout the interface

## v0.14-beta
- Fixed Windows launch crash (WinError 10061) caused by port binding conflict on startup

## v0.13-beta
- Version number now displays automatically in the app title from the release tag
- Added resolution check — files with mismatched resolutions in the same folder are flagged and skipped
- General interface text improvements

## v0.12-beta
- Fixed file discovery — the app now accepts any image filename and extension (JPG, JPEG, PNG, TIF, TIFF), not just files named IMG_*.jpg

## v0.11-beta
- Fixed app crash when relaunching after closing — now uses a random port to avoid conflicts with the previous session

## v0.10-beta
- First public release
- Gradio web interface with folder browse, output folder, frame limit dropdown, and progress bar
- Live status updates during processing
- Opens output folder automatically when done
- Mac (Apple Silicon) and Windows builds
