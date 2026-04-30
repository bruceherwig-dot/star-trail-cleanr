# Star Trail CleanR, Version History

---

## v1.991-beta
- **International users with accented folder names: every read AND every write now handles your paths.** A Slovak tester's run died on the very first frame because the input folder was `C:\Users\magio\Desktop\Štrba\svetlá\` and Windows' OpenCV uses old file APIs that can't open files whose path contains non-ASCII characters (Slovak `Š`, `ľ`, `á`; same problem hits Czech, German, French, Cyrillic, CJK, every European language with diacritics, and CJK languages). The bug fails BEFORE OpenCV even tries to decode the image, which means our v1.99 fallback ladder (cv2 → tifffile → retry) couldn't help — every retry hit the same Unicode-path failure. v1.991 adds Pillow as a third fallback in the chain for reads, AND adds a Pillow fallback for writes (cv2.imwrite has the same Windows Unicode-path bug). Pillow uses Python's normal file APIs which handle Unicode correctly on every platform, so it transparently rescues affected reads and writes. The popup never fires for affected users — the run just works.
- **Reads covered:** main frame load, JPEG EXIF-rotation re-read, hot-pixel-map read, foreground-mask read, mask-painter image and mask loads. All routed through the same fallback ladder (cv2 → tifffile for TIFFs → Pillow → retry).
- **Writes covered:** hot-pixel map, saved trail-detection masks (when "save masks" is on), and the foreground-mask the user paints in the mask editor. All routed through a new Unicode-safe writer (cv2 first, Pillow fallback on failure).
- **PIL fallback now applies EXIF rotation when the caller asks for color.** `cv2.imread` with `IMREAD_COLOR` honors EXIF Orientation on rotated JPEGs; `IMREAD_UNCHANGED` does not. Our PIL fallback matches that behavior using Pillow's `ImageOps.exif_transpose`, so rotated JPEGs from phones/cameras are oriented correctly even when PIL rescued the read instead of OpenCV.
- **Smoke tests:** 7 new tests lock in the rescue behavior for both reads and writes (PIL fallback on JPEG read, grayscale and uint16 write paths, structural checks that every production cv2.imread/imwrite call site routes through the wrapper). Total smoke suite: 128 tests, still under two seconds.

## v1.99-beta
- **One bad image file no longer kills a whole run.** A Windows 11 tester (Warren) was 113 frames into a 266-frame run when batch 7 hit `_DSC0180.tif` and the worker died with a wall of OpenCV TIFF decode errors. The whole batch was lost and the run had to be cancelled. Now: an unreadable file is handled gracefully. The worker tries a second image-decoding library (tifffile, the scientific TIFF library) when OpenCV refuses, and retries up to three times across roughly four seconds (covering brief external-drive hiccups, USB sleep wake-ups, and similar transient I/O blips). Most files in the "OpenCV can't read this" bucket are recovered silently and the user never sees a thing. If all three attempts still fail, instead of crashing, Star Trail CleanR pauses the run and shows a popup naming the bad file with two clear choices: "Skip this frame and continue" (output gets a one-frame gap) or "Stop Run" (graceful exit, partial output preserved). After the first "skip and continue," a second unreadable file auto-stops the run with a final "multiple unreadable files" notice, in case something is wrong with the source folder more broadly. The notice suggests exporting the whole sequence to JPEGs as the simplest workaround.
- **Per-file diagnostic data flows to Sentry for opted-in users.** When the popup fires, a structured warning event is also sent to crash reporting (only if the user opted into crash reports), with the file path, file size, file extension, OS, and the exact error each reader returned. All events are fingerprinted into one Sentry issue so a tester with many bad files doesn't flood the inbox. The popup itself tells the user the diagnostic data has already been sent automatically — but only when crash reporting is actually on; the line is dropped for users who opted out so the message is never untrue.
- **Fix: Windows 11 testers no longer get tagged as Windows 10 in support emails.** Python's `platform.release()` returns the literal string "10" on both Windows 10 and Windows 11 because Microsoft kept the kernel version at 10.0. The support email body, the run summary written to disk, and the Sentry crash-report tag all relied on that string and so all reported "Windows 10" for Windows 11 users. The fix reads the build number from `platform.version()` instead — build 22000 and above means Windows 11. Three call sites updated, with regression tests so this can't quietly come back.
- **The bad-file popup's mailto link carries a pre-filled subject** so the support inbox can filter "unreadable file" reports cleanly.
- **Smoke tests expanded.** Two new test files lock in the v1.99 contracts: the robust image-read fallback ladder (cv2 success, tifffile rescue when cv2 fails, tuple-form diagnosis on hard failure, grayscale handling, and that the worker still uses the wrapper at every production call site) and the Windows 11 build-number detection (release-string mapping, malformed-input fallback, and that the helper is wired into both the support email and the run summary). Total smoke suite is now 121 tests, still under two seconds end-to-end.

## v1.98-beta
- **Fix: v1.97 crashed on launch for fresh installs.** First-time users (and anyone who'd cleared their saved window position) saw the app die before any window appeared, with a "name 'screen' is not defined" error. The first-launch code branch tries to size the window to 90% of the available screen, but one line that's supposed to look up the active screen was missing — so the very next line referenced something that didn't exist. Added the missing lookup. The crash only fires when there's no saved window geometry, which is why CI smoke and existing installs were unaffected; first reports came in the moment a fresh install of v1.97 launched.

## v1.97-beta
- **Fix: cleaning runs no longer crash when the output folder isn't writable.** A Windows tester running 16-bit TIFF output hit a "PermissionError: [Errno 13] Permission denied" mid-run, mid-stack, with no clear explanation. Common causes: the chosen output folder is on a read-only drive, lives inside a OneDrive synced location that holds a sync lock, has a file open in another app (Photoshop, Lightroom, File Explorer's Preview pane), or is restricted by Windows Defender. Two safety nets now: (1) when you click Clean My Stars, the app first tries to create + write a small probe file in the output folder. If that fails, a clear popup tells you Star Trail CleanR cannot write there and suggests picking a different folder, before any work starts. (2) If a write fails mid-run anyway (an app grabs a file lock partway through, OneDrive interferes after the run is going), the worker now exits with a plain-English error message naming the output folder and likely causes, instead of dropping a Python traceback into the log.
- **Star Log header above the run log.** The cleaning page now shows a centered "Star Log" title over the scrolling run log on the left side. The redundant centered status line that used to echo the most recent log entry has been removed (it was just repeating what the log already showed).
- **Warmup heartbeat: astro phrases during the silent AI-load gap.** First-batch cleaning has a 15 to 30 second window after frames finish loading where nothing visible was happening because the AI model is loading and warming up. The Star Log now streams a rotating set of astro phrases ("Studying your stars," "Hunting for trails," "Sweeping the sky," and others) every 2 seconds with animated dots, so the run never looks frozen. The heartbeat starts the moment frame loading finishes and stops as soon as the first frame is detected.
- **Painted X close button.** The red close button's X glyph drifted off-center because font metrics for the multiplication sign character vary by platform. The X is now drawn directly with two diagonal lines, perfectly centered at any size, on every OS.
- **Email moved to a per-app alias.** Every public email link in the app and on the GitHub README now uses bruceherwig+startrailcleanr@gmail.com. Routes to the same inbox but tags the source so app-related mail is easy to filter.
- **About tab and README copy refresh.** Tighter wording about the Claude Code partnership, switched the project framing to "my free gift" instead of "a free gift," replaced the closing line of the Acknowledgments section with a plain "Thank you, all of you," and dropped em dashes from the FAQ workflow steps and other public-facing copy.
- **Project site link added.** "Project site: StarTrailCleanR.com" now sits at the top of the Links list in both the About tab and the GitHub README.
- **Photos for sale link goes straight to the shop.** Visible link still reads "bruceherwig.com" but it now opens the Square shop's astrophotography category directly.
- **Run-complete summary: "across N twinkling frames."** Small whimsy add to the post-run dialog's trail-count line.
- **Sentry crash-report test coverage tightened.** Added regression tests that pin down the GUI's crash-report payload (stdout preview, stderr preview, OS tag, stdout-line buffer) so they cannot silently regress in a future refactor.

## v1.96-beta
- **Fix: TIFF 16-bit output finally works in the shipped app.** v1.91-beta added the fix that was supposed to make 16-bit TIFF output stop crashing, and the source code change was correct. But every release since (v1.91, v1.92, v1.93) shipped a frozen bundle that was missing the tifffile library entirely, so the moment a user picked "TIFF 16-bit" as their output format they hit "ModuleNotFoundError: No module named 'tifffile'" and the worker died. Root cause: the build script told the bundler to include tifffile, but the build server itself didn't have tifffile installed in the first place, so the bundler had nothing to include. Fixed by installing tifffile (and scikit-image, which had the same latent issue for a feature that's currently turned off) on the build server before the bundle is assembled. The first user to pick TIFF 16-bit on v1.93 hit the crash within an hour of release; this is the unblock.
- **Fix: Intel Mac builds run again.** The library that PyTorch uses to share data with NumPy changed how it talks between version 1 and version 2. PyTorch's Intel Mac build was still compiled against the version 1 protocol, but the build server was installing the latest NumPy (now version 2). Result: every Intel Mac build crashed at the moment PyTorch tried to load, with "Numpy is not available." Fix is a one-word constraint added to the install line: install NumPy, but stay on the 1.x line. Apple Silicon's PyTorch was rebuilt for NumPy 2 already, so it never hit this issue, but the same constraint is applied to all platforms for consistency.
- **Bundle smoke test now exercises every output format, not just JPG.** The pre-release smoke test that runs in CI used to only check that the bundled app could process a frame with JPG output. That's why the missing-tifffile bug slipped through three releases. The smoke test now runs the bundled worker once for each of the three output formats (JPG, TIFF 8-bit, TIFF 16-bit) on three synthetic frames (the worker requires a minimum of three for its repair step). If any format fails to load its dependencies, the build job fails and the broken bundle never reaches users.
- **Run Complete dialog copy polish.** Header now reads "Your skies are scrubbed!" (exclamation, not period). Summary line is forced onto two deterministic lines instead of wrapping wherever the window decides. Stacker examples picked up ", etc." in both the Run Complete dialog and the FAQ tab. Summary now reads "trails from your stars" instead of "trails from your skies" (the header still says "skies").
- **(For the curious: v1.94 and v1.95 were tagged but never released.)** v1.94 carried the tifffile bundle fix but its CI failed at a frame-count bug in my new smoke test. v1.95 fixed the smoke test but Intel Mac CI then crashed on the NumPy ABI issue above. v1.96 is the version that actually ships the tifffile fix to all four platforms (Mac Apple Silicon, Mac Intel, Windows, Linux) plus the dialog tweaks accumulated along the way.

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
