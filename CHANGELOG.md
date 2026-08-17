# Star Trail CleanR, Version History

---

## v2.85-beta
- **Fixed: a run could fail immediately with "need >= 3 frames", over and over, on a folder that was perfectly fine.** If your folder held images of more than one size, or you set a frame range or a frame limit, the app worked out which frames to use and then handed the cleaning engine a different list, including the very files it had just decided to skip. The engine took the frames it was pointed at, found them the wrong size, discarded them, and stopped with nothing to clean. It now hands over exactly the list it planned, so this cannot happen. If a batch ever does come up short, the message says why instead of quoting a number.
- **Fixed, and this one was silent: frames could be repaired using the wrong neighbours.** The app orders your frames by the time they were actually taken rather than by filename, which matters when a camera rolls over from IMG_9999 to IMG_0001 or you have merged two memory cards. That order was being quietly reverted to filename order before the run started, so each frame was repaired using the frames next to it in the folder rather than the ones next to it in the sky. No error appeared; the results were just worse.
- **The "remove hot pixels and colored specks" option no longer marks your stars.** It used to erase a dot for being the brightest thing in its neighbourhood, which in a star trail is very often just a bright star, and the way it patched the spots left small dark holes in the trails. Both are gone. A speck is removed only when your own frames prove it, in one of two ways: it sits at the same pixel over and over, which is a fault in the sensor, or it appeared in a single frame and never again, which no star can do, because the sky carries a star's light to a predictable place in the very next frame. On a 300 frame test sequence this took the number of repairs from 2,014 down to 829, removed every speck that could be shown to be a single frame event, and left no dark spots behind.
- **Repairs blend in now.** Each speck is painted out using the picture immediately around it, so a trail running through the spot carries on through it, and no repaired spot is ever left darker than the sky around it.
- **Your clean now remembers where your camera's stuck pixels are.** It works this out while cleaning, twenty frames at a time, and the star trail keeps that record instead of throwing it away, so it knows exactly where to look. Folders you cleaned with earlier versions have no record, so the option works it out from the frames, which adds a few minutes.
- **The star trail built from your original frames now appears in the preview.** It was being created and saved correctly, but the left arrow beside the preview never listed it.
- **Speck removal now judges each dot against the sky around it** rather than a fixed brightness number, which was only ever right for a dark sky. On a brighter sky nearly everything cleared that number, so the option repaired far more than it should have. On a test sequence this cut the repairs from 8,720 to 2,620, while a dark-sky sequence stayed where it was.
- **Smoke tests:** 297 passing.

## v2.84-beta
- **The foreground mask can no longer be painted at the wrong size.** In a folder that mixes image sizes, the mask editor used to open on whatever file sorted first, even if the run would later skip that file for being the odd size out, so a correctly painted mask could be refused with "the foreground mask does not match these frames." The editor now paints on a frame of the majority size, the same one the run keeps.
- **A right-shaped mask at the wrong size now just works.** If your saved mask matches the frames' shape but not their exact size, say you re-exported your images larger, the app scales the mask to fit and notes it in the Star Log instead of stopping the run. A genuinely mismatched mask, portrait against landscape, still stops with the same clear message.
- **Smoke tests:** 277 passing.

## v2.83-beta
- **The app now shows you it's using your graphics card, not just when it isn't.** A small indicator next to the version in the header reads "GPU: NVIDIA" or "GPU: Apple Silicon" in green when your graphics card is doing the work, "CPU only" when there's no card to use, and an amber "GPU: off" when your computer has a card that isn't being used, and clicking that opens Settings where one button fixes it. The finished-run summary confirms it too: "Cleaned using your NVIDIA graphics card." Asked for by a Windows tester who wanted to know before committing to an hours-long run, and we agreed.
- **The Star Trail tab gets the same Source choice as the Timelapse tab.** Build your star trail from the cleaned frames (the default) or from the untouched originals, trails and all, for before-and-after comparisons.
- **Your first star trail now appears in the preview.** If the window opened before any trail existed, a newly built one never showed up. It does now.
- **Browse your builds with arrows.** Make more than one star trail or timelapse and left/right arrows appear beside the preview to flip between them, newest first. Clicking, Open, and Play always act on the one you're looking at, and a caption shows each file's name, which records the settings it was built with, now including whether it came from cleaned or original frames.
- **The Summary tab no longer shows a different folder's numbers.** Switching to a new set of images used to leave the previous run's trail counts on screen.
- **The community stats now count star trails created**, the ones deliberately built on the Star Trail tab, powering the new tile on startrailcleanr.com/stats.html. Settings only, sent anonymously, opt-in as always.
- **Smoke tests:** 272 passing.

## v2.82-beta
- **Checking for updates on Windows works now. It never did before.** Since the one-click updater arrived in May, every check on every Windows PC failed with "an error occurred in retrieving update information," and the app wrongly suggested the user's security software was to blame. The real cause was one line of our code handing the update engine a garbled web address. With it fixed, Windows users get the same one-click updates Mac users have had all along. Getting THIS version still takes one manual download (the copy on your PC has the old bug); every version after arrives with one click. If we told you your security suite or VPN was blocking updates: it wasn't, and we're sorry.
- **Every Windows release build now proves a real update check works before it can ship.** The build machine installs the app, checks our live update feed, and the release is blocked if the check fails. This is the test that found the bug; it is now permanent.
- **Smoke tests:** 266 passing.

## v2.81-beta
- **Photos with extra channels now open instead of failing partway through.** A TIFF saved out of Photoshop can carry more than the usual red, green and blue, because saved selections and masks are stored inside the file as extra channels. Those files got as far as the trail detector and then failed. The app now uses the red, green and blue and ignores the extras, noting it in the Star Log, so the clean runs normally. Your photograph is unaffected; only the stored selections are left out, and they were never carried into the cleaned copy anyway. Black-and-white TIFFs saved with transparency now open too.
- **Fewer false crash reports.** The app tries several different readers when opening a photo and expects some of them to turn it down, but those routine refusals were being reported to us as if the app had crashed. They no longer are. A file that genuinely cannot be opened still reports, with the full detail.
- **Smoke tests:** 261 passing.

## v2.80-beta
- **The app can no longer use your graphics card without telling you, or stop using it in silence.** On a Windows PC with an NVIDIA card, several things could quietly send a clean back to the much slower processor, and the only place it was ever mentioned was a line on the Settings tab. Now, if your computer has a graphics card the app is not using, the run window and the Star Log say so in plain language and tell you how to fix it, and the orange banner offering GPU support comes back if support that used to work stops being used (even if you dismissed that banner long ago).
- **A failed GPU support install can no longer cost you the GPU support you already had.** The installer used to delete your working files before downloading the new ones, so a blocked or interrupted download left you with no graphics acceleration and no explanation. The download now goes to a separate folder and only replaces your working copy once it is complete. If anything goes wrong, you are left exactly where you started, and the message says so.
- **The app will not install GPU support it cannot use.** If the exact version this release needs is unavailable, the installer used to substitute an older one, spend several gigabytes downloading it, and report success, while the app quietly refused to load it at every launch. It now stops before the download and explains why.
- **Checking for updates on Windows no longer ends in a dead-end error box.** If your security software or network blocks the updater, you used to get a message from the update engine itself asking whether you were connected to the internet, with nothing you could do about it. Windows now checks quietly and shows one clear message instead, naming the new version when we know there is one and offering a Download Latest button that fetches the installer in your browser. You can install it straight over your current copy; your settings and folders are kept.
- **Update checks now come from our own server.** The service that told the app about new versions was unreachable for some users, whose security software or network blocked it, leaving them stuck on an old version with an error box. The app now reads update information from startrailcleanr.com instead. Existing copies keep working exactly as before.
- **Smoke tests:** 256 passing.

## v2.79-beta
- **Timelapse and video features now actually ship with their video encoder.** A Windows user's error report (thank you, Philip) revealed that no installed copy of the app has ever contained the high-quality video encoder: it was missing from the build recipe on every platform, and only worked in development where it happened to be installed. Effects until now: rendering a timelapse on an installed copy crashed with an error dialog, and the before/after share clip quietly used a lower-quality backup encoder. Both are fixed: the encoder is now bundled on all platforms, and every build is machine-tested to prove the real encoder is inside before it can be released.
- **A missing encoder can no longer crash the timelapse.** Even if something on a user's machine strips the encoder out (some antivirus software does), the timelapse now falls back to a backup encoder and still produces a movie, with a note in the log, instead of a crash dialog.
- **Removed a leftover experimental sky-speck cleanup from star trail builds.** A June experiment that automatically erased stuck camera pixels from the finished star trail turned out to almost never run (the app deleted its defect map before the star trail was built), and in rare cases it could smudge real scenery it mistook for defects. It's gone. Sky speck cleaning is the "Remove hot pixels & colored specks" checkbox in the Star Trail window, which is unaffected; ground cleaning during the run is unaffected.
- **Smoke tests:** 242 passing.

## v2.78-beta
- **The star trail build's progress bar now tracks the whole job.** It used to fill up within seconds (it only followed the frame-stacking step) and then sit at 100% while the finishing steps, sky speck cleanup, trail thickening, and saving, ran silently for most of the build. Now every step reports its progress, the bar's sections are sized by how long each step actually takes on your computer (it learns from your previous build), and it only reaches 100% when the image is written. The label under the bar names the step it's on.
- **Trail Thickness is much faster.** The thickening step was redoing the same work tens of thousands of times; on a full-resolution build it took about 13 seconds and now takes about 1. The result is pixel-for-pixel identical.
- **The Star Trail preview image is centered.** It was hugging the left edge of the window since v2.76; it now centers like the Timelapse preview.
- **Smoke tests:** 242 passing.

## v2.77-beta
- **When something goes wrong after a run, the app now tells you.** A Windows report on v2.76 hit three problems at once with no error anywhere: the Open Cleaned Folder button did nothing, the Star Trail & Timelapse button did nothing, and the before/after video never appeared. Those failures were happening silently. Now: a button that fails shows a message saying what went wrong, the Star Log always reports whether the star trail and video were built (and if not, the reason in plain language), and the full technical details are saved to a small error log file (.star_trail_cleanr/app_errors.log in your home folder) that you can email us. This release makes the problem visible so it can be fixed for real; if you hit it, please send that file.
- **Smoke tests:** 242 passing.

## v2.76-beta
- **Fixed a slowdown on busy skies.** A change back in v2.66 quietly made cleaning runs with lots of trails per frame (a dense Milky Way, a satellite-heavy night) take far longer than they should. This fixes it, so those runs are back to normal speed. If heavy sequences had been crawling for you lately, they're quick again.
- **Tidier preview and summary in the Star Trail & Timelapse window.** The preview now uses a fixed landscape frame, so a tall portrait shot no longer stretches the window (portrait images center with side margins). This also closes an empty gap that could appear on the Summary tab.
- **Smoke tests:** 242 passing.

## v2.75-beta
- **Comet Mode and hot-pixel removal now work together.** Turning on "Remove hot pixels & colored specks" used to quietly cancel Comet Mode, leaving you with a plain trail instead of a comet. Now you get both: the comet fade is kept and the specks are still removed.
- **The finished-run window no longer pops open a Finder folder.** When a run ended, the STC Extras folder used to open on top of the app. It doesn't anymore. The star trail and timelapse tools open as usual, and the Open Cleaned Folder button is there whenever you want the folder.
- **Smoke tests:** 242 passing.

## v2.74-beta
- **Cleaned sky no longer comes out darker than the sky around it.** On open sky, the repair could leave each cleaned spot a few shades dark, showing up as faint dark streaks where trails used to be. The step that rebuilds dark foreground (tree spikes, rocks) under a trail was mistaking ordinary night sky for foreground and pulling it too dark. It now tells them apart, so cleaned sky matches its surroundings. Dark foreground under a trail is still rebuilt as before.
- **Star trails and timelapses now live in one window.** When a clean finishes, a single window opens with three tabs: Summary, Star Trail, and Timelapse.
  - **Summary:** what was removed, the time you saved, a before/after clip to share, and where your cleaned frames are.
  - **Star Trail:** build a normal (lighten) or comet trail, set the comet length and trail thickness, and remove hot pixels and colored specks.
  - **Timelapse:** choose size, frame rate, and smoothing.
- **The star trail and the before/after clip now build automatically during the clean,** ready the moment it finishes. No checkboxes to remember.
- **Timelapse:** added a 6 fps option, and the options lock while a movie renders so they cannot change mid-render.
- **The Settings tab now scrolls,** so nothing is cut off on smaller screens.
- **Fixed:** the Open Cleaned Folder button on the Summary screen did nothing when clicked. It now opens the folder.
- **Smoke tests:** 242 passing.

## v2.73-beta
- **Crossings now pull in real sky instead of a copy of the crossing trail.** When a trail sits on top of trails in *both* the frame before and the frame after (two trails crossing, or one lingering in the same place for a few frames), there is no clean sky right next door. The repair used to fall back on the nearest neighbor even though it still carried the trail. Now it reaches a frame or two further out, where the trail has usually moved off, and borrows genuinely clean sky, slid so the stars still line up. It falls back to the old behavior only when nothing clean is within reach.
- **Smoke tests:** 241 passing.

## v2.72-beta
- **Repairs next to a tree, ridge, or building no longer come out slightly dark.** When a trail passed close to a dark shape (a trunk, a Joshua tree, a rooftop, a ridgeline), the repaired patch of sky could end up a few shades darker than the sky around it. The cause: the step that brightness-matches the fill was reading that dark foreground as if it were sky, so it nudged the whole fill darker. It now measures only the sky when matching, so the repaired patch blends into the surrounding sky. Applies to every frame, including the first and last of a sequence.
- **Smoke tests:** 241 passing.

## v2.71-beta
- **Trees, rocks, and rooflines survive a trail crossing.** When an airplane or satellite trail crossed a dark foreground shape — a Joshua tree spike, a branch, the edge of a building — the repair could erase part of that shape along with the trail, leaving a chewed-up edge. Now the repair takes advantage of the fact that your foreground sits at the exact same spot in every photo (you're on a tripod) while the trail is bright and moving from frame to frame: it rebuilds those dark shapes from the darkest version of each pixel across nearby frames, so the trail disappears and the silhouette comes back sharp. The sky and stars around it are handled exactly as before.
- **Smoke tests:** 241 passing.

## v2.70-beta
- **Timelapse: choose your own smoothing.** The Create Timelapse window has a new Smoothing pulldown: None, 2, 3, 4, or 5 frames (it was previously fixed at 3). Smoothing blends each movie frame with the previous ones; higher values smooth out flicker, but stretch each star into a slightly longer streak. A caption in the window explains this, and your choice is remembered for next time.
- **Timelapse window polish.** The progress bar now clears when a render finishes instead of sitting at 100%, the headline uses the app's blue, and the row titles are bold and right-justified. The Timelapse window's own version is now 1.1.
- **Smoke tests:** 238 passing.

## v2.69-beta
- **Cleaner repair on the first and last photos of a run.** Those two photos only have a neighbor on one side, and until now a trail crossing a tree, rock, or building there could smear the object's edge in both directions: a small light notch bitten out of the object, or a dark smudge pushed onto the sky right beside it. The repair now checks a second nearby frame and keeps everything that isn't moving exactly in place, and it always leaves dark foreground silhouettes untouched (on a tripod, they never move). Middle frames were already protected and are unchanged.
- **Better run diagnostics behind the scenes.** The developer run log now records exactly how every single trail was repaired, so problems like the one above can be found and fixed from the log alone.
- **Smoke tests:** 238 passing.

## v2.68-beta
- **Update checks are steadier, and they never crash.** In rare cases a connection that dropped partway through checking for a new version could crash the app. That is fixed: a failed check now quietly does nothing.
- **A backup way to find updates when GitHub is blocked.** In some countries and on some networks, GitHub (where the app is hosted) is blocked, which used to leave the app unable to check for or download updates at all. Now, if it cannot reach GitHub, the app checks our own server instead. If a newer version is waiting, it lets you know and points you to startrailcleanr.com to download it by hand. Your photos, settings, and folders are all kept.
- **Smoke tests:** 235 passing.

## v2.67-beta
- **A cleaner finish to every run.** The run summary and the timelapse window now lay themselves out neatly: each one centers on its own, and when both are open they sit side by side instead of overlapping.
- **Create a timelapse right from the run page.** A new "Create Timelapse" button now appears on the run page whenever cleaned frames are available, so you can jump straight to a timelapse without going back to the main screen.
- **Smoke tests:** 230 passing.

## v2.66-beta
- **Cleaner repair where a trail crosses your foreground.** When a plane or satellite trail passed in front of a tree, a hill, a building, or right along the horizon, the old repair could leave a dark notch, a bright smudge, or a small step in the silhouette edge. The repair was rebuilt to handle these crossings far better: it keeps the still parts of your scene (the ground and foreground) exactly in place and slides only the sky to follow the stars, so foreground edges stay clean and the stars stay where they belong. Black holes and bright patches at foreground crossings are gone, and it works the same whether your foreground is dark or bright.
- **Better stuck-pixel cleanup.** Hot or stuck sensor pixels are now caught even when they first show up partway through a sequence, the app builds the list up across the whole run instead of locking it in on the first batch. Ground stuck pixels are patched as it goes, and sky stuck pixels are cleaned once at the end on the finished star trail, with a fill that won't smear your thin trails.
- **Smoke tests:** 230 passing.

## v2.65-beta
- **Windows: a way out when the in-app updater is blocked.** On some Windows machines a security suite, VPN, or firewall blocks the updater's connection, so "Check for Updates" failed with a brief error and left you with no obvious next step. Now, when that happens, Star Trail CleanR explains the likely cause in plain language and gives you a one-click "Download Latest" that opens the right installer in your browser, so a blocked update is never a dead end. The normal one-click update is unchanged for everyone else.
- **Special Thanks now lives on startrailcleanr.com.** The About tab's contributor link and a new FAQ entry, "Who helped make Star Trail CleanR?", open the Special Thanks section on the new website.
- **Smoke tests:** 228 passing.

## v2.64-beta
- **New: turn your frames into a timelapse video.** A "Create a Timelapse Video" checkbox on the main screen pops a Timelapse window at the end of a run, and a new "Create Timelapse" button lets you make one any time a folder already has cleaned frames, no re-cleaning needed. You choose the source (your cleaned frames or the original ones), the size (4K, 2K, or 1080p), the frame rate (12 to 60 fps), and the format (.mp4 or .mov). It shows a size estimate and a free-space check, renders with a live progress bar, and the Render button turns into Stop so you can cancel mid-render. Videos use the same grain-and-bitrate recipe as the before/after share clip, so they hold up if you upload them to Facebook or YouTube. Output lands in the STC Extras folder.
- **Mixed portrait and landscape frames are caught before a run starts.** If a folder had some sideways shots mixed in with level ones, the app used to run a long way in and then stop with an error. It now spots the mismatch up front and tells you plainly how many frames are portrait, so you can pull them out or continue with just the matching ones. (Portrait and landscape each need their own foreground mask, so they have to be cleaned as separate folders.)
- **Renamed the shared outputs** for clarity: the cleaned star trail, the before/after video, and the red trail map now start with "STC_".
- **Anonymous usage data** (opt-in, same toggle) now also notes timelapse settings like size and frame rate, so community stats can include timelapse use. Never your images, file names, or any personal info.
- **Smoke tests:** 228 passing.

## v2.63-beta
- **Frames are now ordered by their actual capture time, not their file names.** If your camera's file numbers rolled over mid-shoot (for example IMG_9999 wrapping back to IMG_0001), or you combined frames from two memory cards, the old name-based order could put a jump in the middle of your star trail and make the cleanup borrow from the wrong neighboring frames. The app now reads each photo's timestamp and processes them in true shooting order. If a folder's photos don't carry timestamps, it falls back to the old name order, so nothing changes for those.
- **Smoke tests:** 216 passing.

## v2.62-beta
- **The time estimate is clearer.** The number up top is now labeled "Initial Estimate," the single best guess made when a run starts, and it stays put. A new "Updated Estimate" sits next to the elapsed and remaining times and refreshes as the run goes, so the figures never contradict each other on a long job.
- **The Star Log stays clean.** Technical warnings from the detection engine no longer appear in the run log on screen, where they looked alarming but meant nothing you needed to do. Only plain, readable status shows now. The full technical detail is still written to the saved log file for troubleshooting.
- **New: optional anonymous usage data, to power community stats.** Star Trail CleanR can now send a small anonymous summary at the end of a run, so we can show community totals on startrailcleanr.com like how many trails have been removed and how many hours saved across everyone using the app. It is strictly opt-in and anonymous: a few counts plus your camera and lens settings read from the photo's own EXIF, and never your images, file names, folders, or any personal information. The first-launch prompt and the Settings toggle now cover both crash reports and this usage data in one choice, so you will see the updated prompt once.
- **Smoke tests:** 211 passing.

## v2.61-beta
- **Fixed the "quit unexpectedly" crash for good.** v2.60 stopped the background star-trail builder when you closed the app, but only the one from the current run. If you ran a clean and then started another, the first run's builder got left behind, and that leftover is what crashed the app the next time you closed it. The app now tracks every builder it starts and stops all of them, both when a new run begins and when you close the app. Your files were never at risk; this was always a quit-time crash, after your cleaned frames were already saved.
- **Your computer no longer falls asleep mid-run.** On a laptop running on battery, the system could idle-sleep partway through a long run and freeze it, sometimes for over an hour, making a 20-minute job look like it took two. The app now keeps the system awake while it cleans. The screen can still dim and sleep to save battery; only the machine itself stays up. Works on Mac, Windows, and Linux.
- **Smoke tests:** 206 passing.

## v2.60-beta
- **Fixed a crash when quitting the app.** Closing Star Trail CleanR could pop "Star Trail CleanR quit unexpectedly" right as it exited. The background helper that builds your star trail and share video *during* a run wasn't being told to stop when you closed the app, so the shutdown collided with it. The app now stops every background task cleanly before it closes. Your files were never at risk — this happened only on the way out, after your cleaned frames were already saved.
- **The FAQ reads as plain questions now** — "How do you find trails in my photographs?", "How do you remove the trails you find?", "What is the workflow?", "What are known limitations?" — with the workflow steps spelled out more clearly. Small label cleanups too (the output setting now reads "JPG quality").
- **Smoke tests:** 203 passing.

## v2.59-beta
- **Your star trail and share video are ready the instant cleaning finishes.** They used to be built in a second pass after the run, which could add minutes on a big job. Now they're assembled in the background *while* your frames are being cleaned, so by the time the run ends they're essentially done — the app finishes them in a second or two and opens the folder with both inside. "Cleaning Complete" now waits for them, so when it says done, everything is actually there.
- **No more stacked update prompts.** If both a new app version and a new Trail DetectoR were available, you could get two orange notices at once with two Download buttons — confusing, and a couple of Windows users hit a dead-end first press. The app update now takes priority (it already includes the newest detector), and when an app update quietly brings a new detector, the app tells you once on the next launch.
- **"Preview Star Trail After Cleaning."** The Output Options star-trail checkbox is renamed to say what it actually does.
- **"Open Me First" instructions in the download.** Mac and Windows each get a short text file explaining how to get past the first-launch security warning, so new users aren't stuck.
- **Smoke tests:** 203 passing.

## v2.58-beta
- **The one-click updater no longer crashes.** Starting in v2.56, the app could quit the moment the updater went to show its window, whether you clicked the update notice or Check for Updates. It was an internal type mismatch in a macOS hook added that version. The hook is removed and the updater shows and installs normally again. Note: if you are on v2.56 or v2.57, you will need to download v2.58 from the website by hand this one time to get past the crash. From v2.58 forward, in-app updating works on its own.
- **The in-app update notice is reliable now.** The orange "new version available" banner could silently fail to appear on some machines, leaving you with no idea an update was waiting. Its check confirmed the secure connection to GitHub using the computer's own certificate store, which the installed app cannot always reach (a VPN, a work network, or certain security settings can block it), and when it failed there was no visible sign. The app now carries its own certificate bundle, so the update check, and the AI model update notice, work reliably no matter how the machine is set up. If your update banner was not showing, this is why.
- **Main screen refresh.** The numbered step titles are now blue to match the FAQ tab, the spacing between steps is more even and roomier, and the intro line under the headline is solid black for stronger contrast. Step 3 now reads "Create a Foreground Mask."
- **Cleaner punctuation across the app text.**
- **Smoke tests:** 193 passing.

## v2.57-beta
- **New: a one-click finished star trail.** A new "Quick and Dirty Star Trail" option in Output Options stacks your cleaned frames into a full-resolution star trail image (`cleaned_star_trail.jpg`) the moment a run finishes — the trail-free composite you'd normally build in StarStaX, made for you automatically. It's a fast lighten-blend (brightest pixel wins), not a polished comet-mode stack, so it's a quick share-ready result, not a replacement for your final edit.
- **The time estimate is smarter and shows up immediately.** "Estimated Time" now appears the moment a run starts instead of waiting for the first frames to be measured, and it accounts for your file type — RAW and 16-bit TIFF take longer to load than JPG, and the estimate now knows that. It also learns your computer's speed from your last run, so repeat runs are tuned to your hardware. It's still a ballpark — a sky full of trails takes longer than an empty one, and we can't know that in advance — but it's a much better first guess.
- **Bigger, cleaner text on the FAQ, About, and Settings tabs.** The body text is a notch larger and the headings, spacing, and margins now match across all three tabs, so they read consistently.
- **"View Star Log" now opens the folder** that holds the log, alongside your run's other files (masks, the star trail, any share video), instead of just opening the log file by itself.
- **When you make share outputs, the folder opens once** — after the last one finishes — instead of popping open for each.
- **Smoke tests:** 189 passing.

## v2.56-beta
- **Star Trail CleanR now creates a shareable before/after comparison video.** After a cleaning run, the app can generate a short looping video with a wipe slider that reveals the before and after side by side. It opens in a folder when it's done. The option is in Output Options.
- **Trail DetectoR v5 is now the detection engine.** Trained on a broader dataset that includes GoPro and action-cam footage, tighter satellite crossings, long-exposure trails, and hard cases the previous version missed. Handles more cameras and trail shapes with fewer blind spots.
- **The comparison video is encoded specifically to survive Facebook's compression.** Facebook re-encodes every video upload through its own pipeline, which is brutal on dark, star-filled skies — banding in the gradients, smeared stars. The clip is encoded to fight that: a fixed high bitrate, dark-scene quantization, and fine grain baked in to protect the sky gradients. Preview quality on Facebook is still lower (that's Facebook's preview, not the file), but the HD view is clean.
- **The update notice now shows reliably inside the app window.** When a new version is available, the native prompt could open behind the main window where you'd never see it. The in-app banner is now the primary notice — it lives inside the window so it can't hide — and it remembers the last update it found so it shows instantly even if the internet check is slow.
- **A spinning indicator in the title bar shows when cleaning is running,** so you can see the app is still working even if you switch to another tab.
- **Run artifacts moved to the cleaned folder.** The Star Log, detection data, and related files now live inside the `cleaned/` output folder (in `STC Extras/`) instead of next to your original photos. Existing folders with the old layout are found automatically.
- **Smoke tests:** 182 passing.

## v2.55-beta
- **RAW photo sets clean reliably now (Canon, Nikon, Sony, Fuji, and more), and the cleaned files keep their capture date.** A run on RAW files could start and then stall partway, with the app stuck open in the Dock and refusing to relaunch. The cause was the detection engine reaching out to the internet mid-run to download a missing add-on the instant it met a file it couldn't read, which only ever happened with RAW. That download is now forbidden outright, so RAW runs finish cleanly. On top of that, each cleaned RAW frame now carries its own capture date and camera, read from the preview built into every RAW file, the same way JPEG and TIFF sources already did. Reported by a tester whose Canon R5 CR3 sets stalled at the end of Step 1.
- **16-bit TIFF output now keeps the capture date too.** Saving to 16-bit TIFF was silently dropping the date and camera for every source, not just RAW, even though that is the format we recommend for preserving full quality. 16-bit TIFF files now carry the capture date, camera, and software stamp like every other format.
- **Smoke tests:** 173 passing.

## v2.54-beta
- **GPU support installs again on Windows.** The one-click GPU install had been failing for every Windows NVIDIA user with a "404 Not Found" error: the download pointed at a file that the PyTorch project never published for the version this build carried. The build is now locked to a version whose files are confirmed to exist, and every future release is automatically blocked from shipping unless its GPU download links are verified live first, so this cannot silently break again. The installer also got smarter on its own: if a file ever disappears from the main server in the future, it falls back to the newest version that does exist and also tries the backup server, instead of giving up on the first error.
- **Silent crashes now leave a trail we can follow.** Some crashes kill the whole app instantly with no error message and no crash report (most often the operating system force-quitting the app when the computer runs critically low on memory). The app now keeps a running flight-recorder log on disk while it cleans: every frame, every stage, and how much memory was left at that moment. If a run ends in one of these silent kills, the very next run automatically folds those last recorded steps into the Star Log, so a single emailed log pinpoints exactly where and why the crash happened.
- **Smoke tests:** 173 passing.

## v2.53-beta
- **RAW photo runs no longer die partway through on ordinary machines.** The pre-run memory check was counting RAW frames at half their true size (a RAW file decodes to a 16-bit image, twice the weight of a JPG at the same resolution), so the app could bite off a bigger batch than the computer could hold and the run would stall with an error mid-way. RAW frames are now counted at their real size, so the app picks a batch that genuinely fits. Reported by a tester running 20 Canon CR3 files.
- **One dotted trail no longer comes out wearing several overlapping detection outlines.** Airplane trails that show as a row of dashes (blinking lights) could fool the crossing detector into seeing two trails where there was one, splitting a single trail into stacked near-parallel pieces. Two new sanity checks fix this: pieces produced by a split must actually point in different directions, and a "second trail direction" inside a blob must carry real evidence, not just a few stray traces across the dashes. Real crossings are provably untouched, including the hardest multi-trail tangles.
- **Detections the false-positive filter rejects now stay rejected everywhere.** Previously a rejected detection was erased from the painted mask but still got cleaned by the repair step and still appeared in exported detection data. The rejection now removes it from both, so the app no longer spends time repairing spots it already decided were not trails.
- **Smoke tests:** 169 passing.

## v2.52-beta
- **Folders with simple numbered names now process in the right order.** If your frames are named without leading zeros (1.jpg, 2.jpg ... 900.jpg, as GoPro and some cameras and phones do), Star Trail CleanR was reading them in text order (1, 10, 100 ... 2, 20), which scrambled the sequence. That made repairs borrow from the wrong neighboring frames and put the final stack out of order. Frames are now sorted by their actual number, so 2 follows 1 and 10 follows 9. Your files are never renamed or changed; only the reading order is fixed. Folders with zero-padded names (IMG_0001, DSC_0001, and similar) were always correct and are unaffected.
- **The low-memory warning no longer cries wolf.** Before a run, the app warned about memory whenever things were tight, even when you actually had more free memory than the job needed. It now only warns when your free memory is genuinely below what the run requires, and a tight-but-sufficient computer simply runs at a smaller batch size with no interruption.
- **Windows auto-updates get the same safety net as Mac.** Every Windows build is now launch-tested before release to confirm its updater actually starts, so a broken updater can't ship unnoticed.
- **Smoke tests:** 169 passing.

## v2.51-beta
- **Automatic updates on Mac actually work now.** The app's built-in one-click updater has been silently failing to start in every Mac version to date — one missing component killed it during launch, with no error shown anywhere. That's why updates never installed themselves and the Check for Updates button could do nothing. The component is fixed, and we watched the full loop run for the first time: open the app, get the native "new version available" window, click Install, and the app updates itself and relaunches. From this version forward, that's how updating works.
- **A broken updater can never ship again.** Every Mac build is now launch-tested automatically before release: the build system starts the freshly built app and verifies the updater engine actually comes alive. If it doesn't, the release fails and never reaches anyone.
- **One update notice, not two.** With the built-in updater now working, its native window handles new-version notices by itself. The orange in-app banner steps back to being the backup: it only appears if the built-in updater isn't available (and on Linux, which updates via the website).
- **No more silent update failures anywhere.** If the updater can't run — most often because the app isn't in the Applications folder — the app now says so plainly and opens the download page instead of doing nothing. Update checks also keep a small diagnostic log, so if anything ever goes wrong again, one file tells the whole story.
- **Main page polish.** The spacing above the folder field now matches the other steps, the blue frame count sits centered over the buttons, a clearer welcome line, and image dimensions read "(6,000 x 4,000px)".
- **FAQ and About refreshed.** The Star Bridge description now matches how the repair really works (color-matched borrowing from the best neighboring frame), the Trail Detection description no longer overpromises, and the About tab gained an Instagram link.
- **Smoke tests:** 164 passing.

## v2.50-beta
- **Tangled crossings now get cleaned.** When three or more trails crossed through the same spot (an airplane crossed by satellites, for example), the app could fail to separate the tangle and would then discard the whole detection as a suspected false alarm — leaving every trail in that crossing untouched in the cleaned photo. Now, when a tangle can't be separated into individual trails, the app recognizes it as a genuine crossing, keeps the AI's detection exactly as found, and repairs all the trails through it. Simple two-trail crossings were always handled and are unchanged.
- **Mac installing and updating made foolproof.** Three related fixes for a problem where macOS silently disables the app's built-in updater if the app isn't properly installed in the Applications folder — which left some users stuck downloading every release from the website by hand:
  - The downloaded disk image now shows the app next to an Applications-folder shortcut, the standard "drag here to install" layout, instead of a lone icon.
  - If the app is launched from the disk image or another temporary location, it now tells you right away that automatic updates are off and how to fix it (drag the app into Applications).
  - The Check for Updates button never silently does nothing anymore. If the built-in updater isn't running, the app explains why and opens the download page instead.
- **Smoke tests:** 164 passing.

## v2.49-beta
- **Folders that mix file types now clean instead of stopping with an error.** If your folder holds both 8-bit and 16-bit versions of the same shots (a common export leftover), the app used to halt and ask you to move one set out. Now it checks every frame up front, picks the depth most of the sequence uses, quietly evens out the odd frames to match, and just cleans. Your originals are never touched, and a one-line note in the run header tells you it happened.
- **Repaired spots now match the sky around them, always.** Three related fixes to how trail gaps are filled:
  - The first and last frames of a run no longer show faint bright or dark rectangles where trails were removed. The sky's brightness drifts slightly at the very start and end of a night, so patches borrowed from a neighboring frame could sit a few shades off. Every patch is now matched to the sky immediately around it before it's pasted.
  - Repairs near changing light — a wildfire glow, drifting smoke, twilight gradients — no longer leave off-color rectangles. When the light genuinely changes between frames, the app now borrows from whichever neighboring frame matches the local color best, and color-corrects every borrowed pixel to the frame being repaired, so a patch can never show a color that wasn't really there.
  - The last-resort fill (used when no clean neighbor exists, e.g. a slow satellite covering the same spot for three frames) now uses real neighboring sky with the same color matching, instead of a synthesized patch that could look speckled on smooth twilight skies.
- **Faint stars survive repairs better.** Repairs used to blend two neighboring frames together, which dimmed the faintest borrowed stars toward the sky. Repairs now borrow from a single neighbor shifted precisely into place, so borrowed stars keep their full brightness.
- **Smoke tests:** 164 passing.

## v2.48-beta
- **Updates now install themselves with one click, the way they're meant to.** When a new version is available, the in-app update notice downloads it, installs it in place, and restarts Star Trail CleanR for you, on Mac and Windows. It no longer sends you to the website to download and reinstall by hand. (Linux still uses the download page, since it has no built-in installer.)
- **The app now checks for a new version the moment you open it, every time.** Previously the built-in installer only checked on a once-a-day schedule, so a new release could take up to a day to show up. Now it checks on launch and offers the update right away when there's something new.
- **Auto-updates are reliable again.** The behind-the-scenes step that publishes updates to people who already have the app had stopped running, so recent versions weren't reaching existing installs automatically. Publishing now happens automatically on every release, and the release fails loudly if it doesn't go through, so updates can't silently stall again.
- **Smoke tests:** 148 passing.

## v2.47-beta
- **Trail gaps are now filled with matching sky color instead of black.** When a trail is removed, the app rebuilds that spot using the real sky and stars from the frames just before and after. In the few places it can't borrow a clean view — most often where a slow satellite sits on almost the same pixels for three frames in a row — it used to drop in a small black patch. Black disappears in a finished star-trail stack, so you'd never see it there, but it showed up as a dark mark in the individual cleaned frames and flickered through any timelapse made from them. Those spots are now painted with the surrounding sky's own color and grain, the real stars around the gap are kept, and the edges are softly blended so the patch melts into the sky instead of punching a black hole. The main repair (borrowing real sky and stars from neighboring frames) is unchanged.
- **Each cleaned frame now keeps its own capture date and time.** The capture timestamp was being copied from the first photo in each batch of 20 onto all 20 frames, so most cleaned files showed the wrong time and the times jumped every 20 frames. Now every cleaned frame carries its own original date, time, exposure, lens, and GPS exactly as the camera recorded them. (Your filenames were always correct — this only affected the time stored inside the file. Applies to RAW, JPEG, and TIFF.)
- **The run-log link now reads "View Star Log (with run detail)"** so it's clearer what it opens.
- **Smoke tests:** 148 passing.

## v2.46-beta
- **Photos shot in portrait (or with the camera turned) no longer come out rotated.** If your camera recorded an orientation tag, the cleaned files were being rotated an extra 90 degrees, so a portrait shot came out sideways. The app now turns every frame upright internally and saves the cleaned files in the same orientation as your originals, for every format. Landscape photos on a tripod were never affected. If you already cleaned a portrait set, just re-run it to get correctly-oriented files.
- **Smoke tests:** 148 passing.

## v2.45-beta
- **You can now pick a clearer photo to trace your foreground mask against.** The mask painter used to always show the first photo in the folder, which sometimes isn't the easiest one to see the skyline in (clouds, headlights, or a bright glow right along the horizon). A new "Skyline hard to see?" control with left/right arrows sits in the green banner at the top, letting you step through your photos and trace against whichever one shows the ground most clearly. It only changes the background picture you're looking at; the mask you paint is shared by every frame, and your painting, zoom, and pan are kept as you switch. (Single-photo folders don't show the arrows.)
- **Smoke tests:** 148 passing.

## v2.44-beta
- **RAW files are now supported.** Drop in a folder of camera RAW files (Canon CR2/CR3, Nikon NEF, Sony ARW, Fujifilm RAF, Adobe DNG, and most others) and Star Trail CleanR processes them directly. No more converting to JPEG or TIFF first. Every frame in the sequence is developed the same way (fixed brightness, the camera's own white balance) so your final stack stays even. Your output choice is unchanged: pick 16-bit TIFF to keep the RAW's full quality, or JPEG for smaller files.
- **If a folder holds both a RAW and a JPEG/TIFF of the same frame**, Star Trail CleanR asks once which to process (RAW by default). A frame that exists in only one format is always kept.
- **RAW works everywhere:** the folder picker, the foreground mask painter, and the previews all read RAW files.
- **Clearer message when a RAW can't be read.** If a RAW file can't be decoded (a very new or unusual variant, or a damaged file), the app now says so plainly and points you to exporting that sequence as 16-bit TIFF or JPEG.
- **Smoke tests:** 148 passing.

## v2.43-beta
- **Very small images are now caught up front instead of crashing.** If you point Star Trail CleanR at downsized previews or web-sized exports (anything under 1280 pixels on the shorter side), it now stops with a clear message asking you to use your full-size originals, instead of failing partway through with a confusing error. Trail detection needs full-resolution frames, so these were never going to give good results anyway. Real camera files are far larger and are completely unaffected.
- **Folders that hold both a JPG and a TIFF of the same photo no longer crash the last batch.** When both versions of a frame were present, the app counted them twice while planning the run, then removed the duplicates later, which could leave the final batch too short and stop with an error. Duplicates are now removed once, up front (keeping the TIFF), so the count is honest, the batches split cleanly, and no frame gets cleaned twice.
- **The up-front "Estimated Time" no longer shows a wildly wrong number.** It used to freeze a first guess based on your previous run, which could read hours off when this run was much faster (for example switching from large TIFFs to quick JPEGs). It now shows "estimating" briefly, then settles on a figure based on the actual measured speed of this run, so the headline matches the live time-remaining.
- **Fewer false internal error reports.** Silenced a harmless font-related message from a background library that was being mistakenly logged as an error. No effect on your runs.
- **Smoke tests:** 143 passing.

## v2.42-beta
- **Large-image runs no longer run out of memory.** On computers with limited RAM, large photos (especially 16-bit TIFFs) could need more memory than the machine had, and the operating system would kill the run partway through with a confusing error. Star Trail CleanR now checks your photos' size and bit depth and your free memory before each run, and automatically processes fewer images at a time when needed so the run fits in memory and finishes. It always picks the largest amount that fits, which is no slower. If a computer is too tight even at the smallest setting, it tells you plainly to close other programs and try again, instead of crashing.
- **New "View Star Log" link.** After a run finishes, is stopped, or hits an error, a "View Star Log" link appears next to the Star Log on the run screen. Click it to open that run's full log, everything that scrolled by plus a run summary, in your text viewer. Handy for your own records or for emailing if something goes wrong.
- **A cancelled run now saves an honest log.** If you stop a run partway, the saved log shows the real progress made so far (frames cleaned, trails found, and time elapsed) and clearly notes the run was cancelled, instead of reporting all zeros.
- **Smoke tests:** 133 passing.

## v2.41-beta
- **The foreground mask now keeps the AI from looking at the ground at all.** Until now the mask only skipped fully-covered areas and cleaned up afterward, so the AI could still detect trail-shaped things on hills, buildings, or equipment wherever they poked into a partly-sky region. Those false hits were removed from the final result but still showed in the mask viewer, which made it look like a mask had fired on the ground. Now the masked foreground is hidden from the AI before it ever looks, so it never detects on the ground in the first place. Your cleaned images are unchanged for real sky trails; the difference is fewer false detections and a mask view that matches what actually gets repaired.
- **Faint "phantom" lines over empty sky are removed.** The AI sometimes drew a thin, dotted line shooting off a real trail into blank sky where nothing is actually there. These are now spotted (thin, faint, with no real streak underneath) and removed before repair, while real trails and real bright crossings are kept. As a bonus this also cleared up some false bridges, since those stray thin pieces were what the bridge step was wrongly grabbing onto.
- **Mixed portrait and landscape batches now stop with a clear message** instead of a confusing error. One foreground mask can't fit both orientations, so run each orientation as its own batch. An empty foreground mask (nothing painted) is now treated as no mask and simply runs.
- **The up-front "Estimated Time" now shows hours** (for example 1h 7m 50s) instead of running the minutes past 60, matching the live time-remaining line.
- **Smoke tests:** 133 passing.

## v2.40-beta
- **Split trails now reconnect even when the AI can't see the gap.** When the AI breaks one trail into two pieces with an empty stretch in the middle, the app already had a bridge that checks six things (matching angle, matching width, the pieces line up, the gap straddles a tile boundary, etc.) to confirm the two pieces are really one trail. Until now, after those checks passed, it would only join the pieces if a fresh look from the AI re-detected something in the gap. On darker, low-contrast sequences the AI sometimes can't see that middle stretch at all, so the trail stayed broken. Now, once the six checks confirm it's one trail, the pieces are joined regardless, the checks are the decision. The result is fewer broken trails in the cleaned output, especially on faint sequences. Trails that aren't really one piece (and static foreground objects) fail the six checks and are left alone, so nothing gets joined that shouldn't be.
- **Smoke tests:** 133 passing.

## v2.39-beta
- **Diagonal and curved trails now get a tight outline.** The fitter that handles long and curved trails was measuring a trail's thickness as its up-and-down height in each slice of the image. On a steep diagonal or a curve that reads far thicker than the trail really is, so those outlines ballooned into wide bands. It now measures thickness straight across the trail at its true angle, so the outline hugs the trail no matter which way it runs. Straight trails are unchanged.
- **One outline per trail.** When the AI produced extra overlapping pieces sitting on top of a trail another outline already covered, those redundant pieces are now folded in. Each trail shows a single clean outline and gets repaired once instead of several times, which means cleaner repairs and a little less work per frame. Crossing trails and genuinely separate trails are untouched.
- **More reliable gap bridging.** When one trail gets split into two pieces, the step that reconnects them was wrongly rejecting some clearly-straight trails (it judged alignment from the far center of each piece, which exaggerates a tiny angle difference on a long trail). It now judges alignment right at the gap, so straight trails reconnect properly.
- **Smoke tests:** 133 passing.

## v2.38-beta
- **Tighter trail outlines.** The detection outlines now hug each trail more closely, grabbing about 24% less surrounding sky. The thickness (across the trail) and the length (along it) are tuned independently, and both scale with your image so they behave the same at any resolution. The result is a cleaner repair with less chance of disturbing nearby stars.
- **New run stats.** The Run screen now shows two live figures under the trail counter: the average number of trails per frame and the average seconds per frame, so you can see how busy your sky is and how fast the run is going.
- **Slightly faster repair.** A redundant step was removed when repairing each trail, trimming a little time off every frame.
- **Smoke tests:** 133 passing.

## v2.37-beta
- **Faster detection.** The step that fits trail outlines now works on a small cropped region around each detection instead of scanning the whole image for every detection. The results are identical, but that step runs about 3x faster, shaving roughly a minute off a typical run.
- **Clearer progress.** The Run screen now names exactly which frames are being detected and repaired, with the whole-job total, for example "Detecting frames 21-40 (of 450)". The numbers no longer get cut off on large jobs and the progress bars stay put.
- **Smoke tests:** 133 passing.

## v2.36-beta
- **Improvement: Detection is significantly faster on images with a visible horizon or foreground.** Star Trail CleanR now skips AI inference on tiles that are entirely in the foreground — trees, buildings, ground — where trails can never appear. On sequences with a clear horizon in the lower third of the frame, this cuts the detection time per frame by 30 to 60 percent. Sequences that are mostly sky (very low horizon) are unaffected.
- **Improvement: Trail-crossing detection is faster.** The step that separates two crossing trails from a single merged detection now runs 4x faster. It uses a cropped region instead of scanning the full frame, and processes pixels as a batch instead of one at a time.
- **Improvement: Elongation filtering is faster.** The step that rejects blob-shaped detections that are too round to be a trail now runs 2x faster using the same cropping approach.
- **Fix: Gap between trail fragments at tile boundaries is now reliably bridged.** The angle tolerance was tightened and two new geometric checks were added — one confirming the gap closes in the direction the trail is traveling, and one confirming the gap actually straddles a tile boundary. Fixes cases where a trail split at a corner would not be re-joined.
- **Improvement: Time estimate now shows hours, minutes, and seconds** instead of rounded minutes only.
- **Smoke tests:** 133 passing.

## v2.35-beta
- **Improvement: Running time estimate and elapsed time are now shown in minutes and hours only.** The "Time remaining" and "Time elapsed" displays in the Run screen no longer show seconds. Short jobs show "5 min"; longer jobs show "1 hr 46 min".
- **Improvement: Folder selection now shows image dimensions.** After you select a folder, the frame count label now includes the width and height of your images, for example "206 frames found (6,720px x 4,480px)".
- **Improvement: In-depth plain-English documentation added to all pipeline modules.** Each major source file now has a full description of what it does, why, and how the key steps work. Intended for anyone reading the code for the first time.
- **Smoke tests:** 133 passing.

## v2.34-beta
- **Improvement: Static false-positive suppressor is significantly faster.** The suppressor now pre-computes per-component bounding boxes for every frame before comparing, then works entirely within each component's bounding box instead of scanning the full image. On a 40-frame batch of 6000x4000 images, the suppressor step dropped from 36 seconds to under 6 seconds.
- **Smoke tests:** 133 passing.

## v2.33-beta
- **Fix: Download button in the update banner now opens the browser on Windows.** The button was silently doing nothing on Windows due to a platform compatibility issue. It now uses the same method as the pre-launch update prompt, which works correctly on all platforms.
- **Smoke tests:** 133 passing.

## v2.32-beta
- **Fix: Two parallel airplane trails in the same frame are no longer merged into one fat polygon.** When two trails run close together (roughly 40 pixels apart), the grouper was treating them as one trail because their detection masks touched. The merging threshold has been tightened so each trail gets its own polygon.
- **Fix: Curved trails are now traced with multiple fitted segments instead of one straight rectangle.** When a trail curves significantly across the frame (spanning more than 1,500 pixels with at least 5 degrees of angle change), Star Trail CleanR now divides it into overlapping strip segments and fits each one separately. This prevents the single-rectangle fit from cutting across stars on either side of the curve.
- **Smoke tests:** 133 passing.

## v2.31-beta
- **Fix: GPU installation no longer fails with a misleading error on networks that block pytorch.org.** When pytorch.org returns a 403 Forbidden error, Star Trail CleanR now automatically retries the download from a backup server. If both servers are blocked, the error message now explains the situation clearly and suggests connecting to a VPN, with a More Info button linking to step-by-step instructions.
- **Fix: GPU installation error dialog no longer gives wrong advice.** The previous dialog told users to check their internet connection and free disk space regardless of what actually went wrong. The message now shows only the specific reason for the failure.
- **Fix: Batch processing no longer drops the last frame of each batch.** A frame-counting bug was causing the final frame in every batch to be skipped during the trail cleaning pass.
- **Fix: Mask output folder moved.** Detection masks now save to the `cleanr_workspace` folder inside your image folder instead of the `cleaned` folder, keeping all intermediate files in one place.
- **New: Static false-positive suppressor.** Detections that appear in the same location across multiple frames are now automatically suppressed. Stationary bright objects — hot pixels, fixed reflections — that the AI occasionally mistakes for trails are filtered out before repair, so they are no longer blacked out.
- **Improvement: Suppressor now also vetoes detections with bright, non-moving pixels.** A detection where the brightest pixels are fixed across frames, or where the centroid does not move between frames, is treated as a static artifact and skipped.
- **Improvement: Edge rescue is more aggressive.** Trails that pass within 20 pixels of the image edge (up from 5) are now pulled in from the edge and included in the detection.
- **Improvement: Trail grouper minimum area raised.** Small stray detections below 1000 pixels are no longer grouped with larger trails, reducing polygon noise on faint detections.
- **Smoke tests:** 133 passing.

## v2.30-beta
- **Fix: App no longer hangs on first launch on Windows.** A crash-reporting consent question was appearing hidden behind the startup screen, leaving the app waiting for an answer no one could see. The app now loads fully first, then asks the question after the main window is open.
- **Fix: Version history link in About now shows the complete history.** The link was pointing to a branch that only had history up to v1.995.
- **Fix: Trail repair no longer causes solid black rectangles on multi-frame trails.** An offset bug was applying neighbor masks at the wrong positions, causing large areas of sky to be blacked out instead of repaired.
- **Fix: Polygon centering on trails that cross tile boundaries.** When a trail crossed a tile seam, the detection polygon was landing between two detection bands instead of on the actual trail pixels. Polygons now center on the real trail pixels regardless of how the trail was split across tiles.
- **Fix: Polygon length no longer clips when one detection is near the size threshold.** A detection just barely pruned by the width filter was excluding its pixels from the polygon extent calculation, chopping the end off the polygon. All detections now contribute to the polygon length.

## v2.29-beta
- **Fix: Download button in the update banner now links to the correct file.** The Mac download URL was pointing to a .zip that does not exist on the release. It now correctly links to the .dmg installer.

## v2.28-beta
- **Fix: Trail repair no longer blacks out entire components on multi-frame trails.** When a trail spans adjacent frames, the repair now uses sky pixels from all neighbor frames and blacks out only the pixels where a neighbor's own trail mask overlaps. Previously, any mask overlap above a threshold caused the entire component to be skipped and filled with black.
- **Fix: Red nav-light trails with diluted color are now detected correctly.** The red channel check now uses the brightest pixels in the detection mask instead of the average. Masks that include surrounding dark sky were dragging the mean down and causing real nav-light trails to be filtered out.
- **Fix: Splitting a combined trail blob no longer discards a valid single-trail detection.** When two parallel trails overlap in a tile seam zone, the splitter now checks whether both halves are trail-shaped before committing to the split. If one half is a fat blob, it reverts to the original detection.
- **Fix: Polygon tips are now tighter and polygon width is closer to the actual trail.** End-cap padding and perpendicular width were both trimmed to reduce overshoot into surrounding sky.
- **Fix: Short diagonal trails are no longer dropped.** The minimum aspect ratio gate was lowered so trails at wider angles pass through.
- **New: TileFixR Cleaned mode.** A new radio button in TileFixR lets you switch to a view-only mode showing the cleaned tiles with mask contours overlaid. No editing is possible in this mode.
- **Smoke tests:** 133 passing.

## v2.27-beta
- **New: Trail DetectoR v4 is now built in.** The AI model has been retrained on a larger, more diverse dataset. It finds more trails, especially faint ones and those near the edges of the frame. Detection accuracy (mAP50) improved from 0.805 to 0.859.
- **Fix: Trails at tile boundaries are no longer missed.** Star Trail CleanR divides each frame into tiles for processing. A suppression bug was causing detections near tile seams to cancel each other out, leaving gaps in the mask. This is now fixed.
- **Fix: Crossing trails are now detected separately more reliably.** When two airplane trails cross at a shallow angle, they can appear as one connected blob. The splitter that handles this case now works correctly for crossing angles and complex intersections that previously caused one trail to be missed entirely.
- **Fix: Closely spaced parallel trails are no longer merged into one polygon.** Two trails flying close together now each get their own mask instead of being combined.
- **Fix: Small red airplane nav-light blobs are no longer filtered out.** Short red trail segments — typical of a flashing port nav light — were being blocked by a size gate. They now pass through regardless of length as long as the underlying pixels are red.
- **Fix: Run screen now shows the dataset folder and its parent.** When your images are in a folder with a generic name like "TIFF," the run screen now shows "Dataset Name/TIFF" so you can see at a glance which set is being cleaned.
- **Fix: Mask painter always opens in Paint mode.** Previously, if you switched to Erase mode and closed the foreground mask editor, it would reopen in Erase mode next time. It now resets to Paint mode on every open.
- **Smoke tests:** 133 passing.

## v2.26-beta
- **Fix: 8-bit TIFF files are now uncompressed.** Same fix as v2.25-beta for 16-bit — all TIFF output is now uncompressed and compatible with Sequator and other stacking apps that require uncompressed TIFF.
- **Smoke tests:** 133 passing.

## v2.25-beta
- **Fix: 16-bit TIFF files are now uncompressed.** Some stacking software (including Sequator) requires uncompressed TIFF and would reject the files Star Trail CleanR produced. The 16-bit TIFF option now writes uncompressed files, which are compatible with all stacking apps. Files will be somewhat larger on disk.
- **Smoke tests:** 133 passing.

## v2.24-beta
- **New: A fitted polygon layer now sits on top of the YOLO detection.** After the AI finds a trail, Star Trail CleanR fits a tight rectangle to the detected trail pixels — closer-fitting ends, more accurate width, and fragments from the same trail joined into one shape. The repair now works from this fitted polygon instead of the raw AI mask blob, which reduces bleed into surrounding sky.
- **Fix: Repair no longer imports trail from a neighboring frame.** When Star Trail CleanR fills a masked area using pixels from the frames before and after, it now checks whether those imported pixels are themselves trail-bright. Any that are get zeroed out instead of pasted in, leaving clean sky.
- **Fix: GPU error message now says to reboot instead of "Clear GPU Support Files."**
- **Fix: Frame limit field now accepts any number you type, not just preset values from the dropdown.**
- **Smoke tests:** 133 passing.

## v2.23-beta
- **Fix: App no longer crashes on image sets that include TIFF files with an embedded alpha channel.** Some software exports TIFFs with a transparency layer that Star Trail CleanR doesn't need. The app now strips the alpha automatically and notes it in the Star Log, so the run continues without interruption.
- **Smoke tests:** 133 passing.

## v2.22-beta
- **Fix: App no longer crashes on incompatible NVIDIA GPUs (Windows).** Some GPU cards aren't supported by the PyTorch CUDA build we ship. Previously this caused a crash on startup. The app now tests CUDA before committing to it, falls back to CPU automatically, and shows an honest status in Settings: "NVIDIA GPU detected but your card isn't supported by the current GPU pack — running on CPU." When a compatible GPU pack becomes available, the app picks it up automatically on next launch with no action required.
- **Smoke tests:** 132 passing.

## v2.21-beta
- **Fix: Finder Comments now write correctly on Mac.** Cleaned files now show the Star Trail CleanR stamp in the Comments field in macOS Get Info. The previous approach was unreliable; it now uses the same mechanism Finder uses internally.
- **Fix: Friendly message when fewer than 3 frames are selected.** Instead of crashing silently, the app now shows a clear dialog explaining the minimum and noting that Star Trail CleanR works on individual frames before stacking, not a finished star trail image.
- **Tweak: "Warming up the AI trail detector" wording cleaned up.** Removed the word "Still" from the fallback warmup message in the Star Log.
- **Smoke tests:** 132 passing.

## v2.20-beta
- **New: Run details appear at the top of the Star Log when a run starts.** The log now shows which AI model is running, which processor (Apple Silicon, CPU, or GPU), the output format, and whether Second ScrubbeR is on — so you have the full picture without digging through Settings.
- **New: The "Email Me" support link includes your Star Log, version, and OS automatically.** Clicking the link opens a pre-filled email with the last 1,500 characters of your Star Log, your app version, and your OS. Much easier to get the right information for a bug report.
- **Fix: Low memory warning now leads with the right advice.** The warning now tells you to close other open programs first, then reduce the number of images only if that isn't enough. The old message led with reducing frames, which most people don't want to hear.
- **Smoke tests:** 132 passing.

## v2.19-beta
- **Fix: Saving as 8-bit TIFF no longer crashes when source files are TIFFs.** The error (RuntimeError: Error setting from dictionary) happened because reading EXIF from a TIFF source pulled in image-structure tags that the TIFF writer already owns internally. The fix strips those tags before writing and adds a safety-net fallback — the same multi-step approach used for JPEG EXIF since v2.09. Camera metadata (make, model, lens, ISO, GPS, date) is preserved intact.
- **New: Low disk space warning before a run starts.** Star Trail CleanR estimates how much space the cleaned output will need based on your first source frame, checks the output drive, and warns you if it looks tight. You can cancel and pick a different output folder, or continue anyway.
- **Splash screen status text is larger and easier to read.**
- **Smoke tests:** 132 passing.

## v2.18-beta
- **Fix: Progress bar stays "Cancelled" after you cancel a run.** Previously the bar could flicker back to a stale percentage while the subprocess was finishing up. It now locks the moment you click Cancel.
- **Fix: GPU installer now shows which files blocked cleanup.** If a previous GPU install left locked files behind, the error message now names the specific files Windows couldn't remove instead of giving a generic message.
- **Fix: Background worker no longer hangs if a dialog is force-closed.** Two dialogs (mixed-resolution warning, unreadable file prompt) waited indefinitely for user input with no timeout. They now release the worker after 5 minutes if no response, stopping the run gracefully.
- **Smoke tests:** 132 passing.

## v2.17-beta
- **GPU installer now shows download progress on the Main screen (Windows).** When you click Install from the orange banner, the banner transforms in place to show a live download progress bar. No need to switch to the Settings tab to see what's happening. Settings still shows progress too for users who install from there.
- **Fix: Orange GPU banner no longer reappears after a successful install.** After installing GPU support and restarting, the "NVIDIA GPU detected" banner now stays gone — it only appears when GPU support genuinely isn't installed yet.
- **Smoke tests:** 132 passing.

## v2.16-beta
- **Fix: GPU installer no longer crashes on large downloads (Windows).** The progress signal used a 32-bit integer that overflows at 2.1 GB. The PyTorch CUDA wheel is 2.75 GB, so the app crashed every time partway through the download. Fixed — the installer now completes the full download without crashing.
- **Smoke tests:** 132 passing.

## v2.15-beta
- **Fix: Finder Comments now actually appear on Mac.** Cleaned files have always been stamped with the Star Trail CleanR version and website in the EXIF Software tag. The Finder Comments field (Get Info) was supposed to show the same stamp but was always blank. The previous code wrote plain text, which Finder ignores — the fix writes the binary plist format Finder expects.
- **Settings tab spacing fixed.** Each section had a large blank gap below its heading. Replaced the fixed-height text boxes that caused it with auto-sizing labels — spacing is now even across all four sections.
- **Checkbox alignment.** Second ScrubbeR and Crash Reporting checkboxes now have matching left indent.
- **Second ScrubbeR locks during a run.** The checkbox grays out and shows a message while a run is in progress so you can't change it mid-job.
- **Tabs now fill edge to edge.** The four tabs stretch evenly to fill the window width. The first tab no longer disappears when the window is narrow.
- **GPU installer cleanup is more reliable (Windows).** Before installing GPU support, leftover files from a previous attempt are cleared with a 3-attempt retry loop that handles read-only files and Windows antivirus locks. If cleanup still fails, the error now tells you exactly what to do.
- **New: Clear GPU Support Files button (Windows).** In the GPU Acceleration section of Settings — removes all GPU support files so you can start a clean install. If it can't delete everything, it shows you the folder path to delete manually.
- **Smoke tests:** 132 passing.

## v2.14-beta
- **Settings: Crash Reporting toggle.** A new Crash Reporting section in Settings shows whether anonymous crash reporting is turned on or off, and lets you change it at any time. The choice you made at first launch is reflected immediately.
- **Smoke tests:** 132 passing.

## v2.13-beta
- **Fix: GPU support retry now works after a failed install (Windows).** If the first GPU support installation attempt failed partway through, clicking Install again would hit a "Permission denied" error because Windows locked the partially-extracted files. The installer now clears any leftover files before each attempt, so retrying always starts clean.
- **Clearer GPU installation error messages.** A Windows permission error now explains what happened and what to do ("Try clicking Install again"), instead of incorrectly suggesting an internet or disk space problem.
- **Cleaned output files now appear in macOS Finder Comments.** After cleaning, each output file's Finder Comments field (visible in Get Info) shows the Star Trail CleanR version, Trail DetectoR version, and website — the same stamp already written to the EXIF Software tag. Skipped if the file already has a comment. Mac only.
- **Settings polish.** GPU Acceleration section no longer shows a redundant description sentence — the status line below it says the same thing. Mac Intel and Linux users see "CPU processing only — GPU acceleration not available on this device" instead of the ambiguous "CPU — no GPU acceleration."
- **FAQ wording.** "The result is a clean set of frames you can stack into a perfect star trail composite" softened to "...star trail composite. (That's the goal, anyway.)"
- **Smoke tests:** 132 passing.

## v2.12-beta
- **In-app GPU support installer (Windows).** When Star Trail CleanR detects an NVIDIA GPU and GPU support is not yet installed, a prompt appears offering to install it automatically. Clicking Install downloads approximately 3-4 GB from pytorch.org, extracts it into a permanent folder the app installer never touches, and asks you to restart. After restarting, the trail detector runs on your GPU automatically. This is a one-time setup — every future app update picks up GPU support without needing to reinstall it.
- **Fix: GPU support button now works.** The "Install GPU Support" button in Settings (and the banner that appears when an NVIDIA GPU is detected) previously opened the GitHub releases page, which had no NVIDIA download on it. The button now launches the in-app installer described above.
- **Smoke tests:** 132 passing.

## v2.11-beta
- **Settings polish.** "Compute Device" renamed to "GPU Acceleration." "Second Scrub" renamed to "Second ScrubbeR" throughout Settings. "Trail Detector" updated to "Trail DetectoR" in the header, run log, and Settings.
- **Second ScrubbeR trail count fix.** Trails found during the Second ScrubbeR pass are now included in the Trails Detected counter in the upper right. Previously only first-pass trails were counted.
- **Smoke tests:** 132 passing.

## v2.10-beta
- **New: Second Scrub option in Settings.** Runs the trail detector a second time on each frame after rotating it 180 degrees, then merges any newly found trails into the repair pass. Catches trails the first pass tends to miss, especially those at angles the detector underweights. Detection takes roughly twice as long — repair time is unchanged. Turn it on in Settings. Works on CPU and GPU builds. If the second pass fails for any reason, the run continues normally on first-pass results with a warning in the log.
- **Smoke tests:** 132 passing.

## v2.09-beta
- **Crash fix: large camera EXIF no longer kills the save step.** Some cameras (especially Sony and certain Canon bodies) write very large blocks of manufacturer-specific data into their EXIF. When that data was close to JPEG's built-in size limit and we added our Software stamp on top, the save failed with a crash. The fix trims only the unreadable manufacturer blob if needed — all the meaningful metadata (camera model, lens, f-stop, shutter speed, ISO, timestamps, GPS) is preserved. If somehow even that isn't enough, the file saves cleanly without EXIF rather than crashing. Original source files are never affected.
- **Smoke tests:** 132 passing.

## v2.08-beta
- **GPU pack survives updates (Windows).** NVIDIA GPU users previously had to redo the 4 GB CUDA swap after every Star Trail CleanR update, because the installer overwrote the app folder. The GPU files now live in a permanent folder the installer never touches. Set it up once, and every future update continues to use your GPU automatically. If a future release requires a different PyTorch version, the app detects the mismatch, falls back to CPU, and tells you exactly what to do in Settings.
- **Smoke tests:** 132 passing.

## v2.07-beta
- **Trail counter now updates live during detection.** The trails-detected count on the Run page now ticks up frame by frame as the AI scans each image, instead of jumping at the end of each batch after repair. The counter shows in blue while the run is active and turns green when the run finishes.
- **Run log renamed and expanded.** The file saved to your `cleanr_workspace` folder after each run is now named `star_trail_cleanr_log_date_time.txt` (previously `run_summary_...`). It now includes a Camera Info section near the top — camera make and model, lens, date and time the sequence was shot, f-stop, and ISO — pulled automatically from your image files' EXIF metadata.
- **TIFF output now carries full camera metadata.** When cleaning TIFF source files and saving as JPEG or TIFF, the output now correctly inherits the original camera data (make, model, lens, exposure, GPS, etc.). A gap in how the app read EXIF from TIFF files meant this metadata was being lost on TIFF inputs; JPEG inputs were unaffected.
- **Crash fix: corrupt or missing mask file no longer kills the run.** One tester's run failed immediately on batch 1 with no useful error message because a foreground mask file had become unreadable since it was saved. The app now catches this before the run starts and shows a clear dialog: proceed without the mask, or cancel and re-draw it.
- **Settings tab: Compute Device section.** A new section in the Settings tab shows what the AI trail detector is running on — Apple MPS on Apple Silicon Macs, NVIDIA CUDA on the GPU build, or CPU otherwise. Lets you confirm at a glance that GPU acceleration is active.
- **NVIDIA GPU detected: upgrade path now built in.** Warren Hatch confirmed the NVIDIA-Accelerated Build cuts processing time in half on large files. When the app detects an NVIDIA GPU on Windows but you're running the standard build, a Download button now appears in Settings (and in the startup banner) linking directly to the NVIDIA-Accelerated Build. The banner text has been updated to reflect this — it no longer says "coming in a future update."
- **Smoke tests:** 132 passing.

## v2.06-beta
- **Settings tab.** A new Settings tab sits between FAQ and About. The first setting is a "Check for Updates" button that triggers the native update dialog on Mac and Windows (or opens the website on Linux). GPU settings will be added here in a future release.
- **Run summary always fires.** Running a clean batch with no trails found used to produce no end-of-run popup. Now the summary appears regardless.
- **Switching macOS Light/Dark mode no longer cancels a running batch.** The mode switch triggers an app relaunch to rebuild the themed UI; that relaunch now waits until any active run finishes.
- **No startup splash on theme-switch relaunch.** When the app relaunches after a Light/Dark mode switch, it skips the startup splash since the app was already open and imports are cached.
- **Warmup phrases no longer repeat back to back.** The Star Log warmup rotation (shown while the trail detector loads) could show the same phrase twice in a row across consecutive batches. It now always advances to the next phrase.

## v2.05-beta
- **First release with delta updates.** Past releases forced every user to redownload the entire ~600 MB app on every update, even when only a few Python files actually changed. v2.05-beta is the first release published with binary delta files alongside the full bundle. Users on a recent prior version (currently v2.04-beta) only download the bytes that actually changed since their version, typically a small fraction of the full 600 MB. Sparkle handles the delta application transparently. New first-time installs still download the full bundle. Mac only — WinSparkle has no delta-update support, so Windows continues to ship the full bundle each release.
- **Update popup no longer hides behind the splash screen.** v2.03-beta exposed a layering issue: when the startup splash was still on screen and Sparkle's "update available" popup appeared, the popup landed *behind* the splash because the splash claims stay-on-top. v2.05-beta wires a Sparkle delegate that fires the moment a valid update is found, dismissing the splash early so the popup is the only thing on screen.
- **No functional changes** to trail detection, repair, output, or other behavior.

## v2.04-beta
- **Auto-update install no longer fails with "improperly signed and could not be validated."** v2.03-beta finally got the update popup working (the App Translocation fix landed), but clicking Install produced an "Update Error" dialog. Sparkle's logs revealed the new app's outer code-sign seal was broken: PyInstaller ad-hoc-signs the bundle on Apple Silicon, then the build script copies in `Sparkle.framework` and patches the Info.plist, both of which invalidate the seal. Sparkle 2.x has a strict safety check — if the new bundle claims to be code-signed but the signature is corrupt, the update is rejected outright. v2.04-beta re-applies a fresh ad-hoc signature on the outer `.app` after all build modifications, so the seal is intact when Sparkle inspects it.
- **No functional changes** to trail detection, repair, output, GUI, or other behavior. v2.04-beta is the second half of the auto-update fix that started in v2.03-beta.

## v2.03-beta
- **Mac builds now ship as DMG instead of ZIP.** Diagnostic logging in v2.02-beta confirmed the v2.0/v2.01/v2.02 auto-update silent-fail was caused by macOS App Translocation: when a `.app` is unzipped from a downloaded `.zip` and launched without first being moved to `/Applications`, macOS runs the app from a randomized read-only path under `/private/var/folders/.../AppTranslocation/...`. Sparkle correctly detects the translocation and refuses to operate (it can't write back to a read-only path to install an update). The fix is the canonical macOS install flow: ship a `.dmg`, user mounts it, drags the app to `/Applications`, launches from there. macOS does not translocate apps installed from a DMG into a known location. v2.03-beta is the first release that uses DMG end-to-end.
- **First real chance for the auto-update flow to work.** v2.0-beta installed the Sparkle infrastructure but couldn't deliver an update because of translocation. v2.01-beta and v2.02-beta hit the same wall. v2.03-beta breaks the cycle. v2.02-beta users (installed from `/Applications`) launching the app should see the native update popup advertising v2.03-beta within seconds.
- **No functional changes** to trail detection, repair, output, GUI, or other behavior.

## v2.02-beta
- **Diagnostic logging for the auto-update flow.** v2.01-beta was supposed to be the first auto-update test (v2.0-beta installs would see a popup advertising v2.01-beta and install it in-app). The popup never appeared. PyInstaller's `--windowed` mode on Mac swallows Python stderr, so the silent-fail mode was invisible. v2.02-beta adds step-by-step file-based diagnostic logging to `~/.star_trail_cleanr/sparkle_debug.log` covering every stage of Sparkle initialization (framework path lookup, PyObjC import, `objc.loadBundle`, the `SPUStandardUpdaterController` init call, exceptions, and update-check invocations). Read the file after launching to see exactly which step failed.
- **Splash status text refinements.** The splash now shows three messages instead of four: "Initializing…" → "Checking for updates…" → "Warming up the trail detector…". Dropped the "Loading components…" message because the phase it described (Sentry SDK init) finishes in 100-300 ms, faster than the eye can read.
- **Mask painter banner copy update.** Added the *why* to the masking instruction: "You're just marking areas where you know trails won't appear, **so the AI doesn't try to 'fix' the ground.**"
- **No functional changes** to trail detection, repair, or output.

## v2.01-beta
- **First real test of the auto-update flow.** v2.0-beta installed Sparkle (Mac) and WinSparkle (Windows) but the appcast feed was empty, so no update dialog ever fired. With v2.01-beta tagged and signed, the feed now advertises v2.01-beta as the newest version. v2.0-beta installs running the app should see a native update dialog on next launch (or within 24 hours if already running) saying v2.01-beta is available. Click Install, the app downloads only the changed parts, and restarts itself into v2.01-beta. **If it works as designed, every future release from here is in-app.** If anything misbehaves we'll catch it in Sentry or your support inbox and fix it.
- **Startup splash screen.** First launches used to show a frozen-looking GUI for a few seconds while the app warmed up (PyTorch imports, model paths, theme detection, Sparkle init). Now you see a splash window with the STC logo, tagline, `#StarTrailCleanR`, and an animated progress bar while the app loads. Brand-navy bars top and bottom match the main app banner. Stays visible for at least 5 seconds, longer if startup runs longer.
- **UI polish on Setup and Run pages:**
  - Run page community panel (the share-message + email link box on the right): now sits roughly vertically centered in its column at any window height, instead of pinned to the top.
  - Setup page "N frames found" label: moved to its own row above the input field, width-matched to the Browse + Open Folder buttons. Stays centered above them as the window resizes. Replaces the prior fixed-padding workaround that drifted off-center when the window grew.
- **No functional changes** to trail detection, repair, or output. Pure UI polish plus the auto-update test payload.

## v2.0-beta
- **First release with auto-update infrastructure built in.** The app now ships with Sparkle (Mac) and WinSparkle (Windows) embedded. The intent: when a future release is published, the app fetches a small XML feed in the background and pops a native update dialog asking if you want to install. **This is a beta, the auto-update flow has not yet been exercised end-to-end with real artifacts.** The first real test happens when the next release after v2.0-beta ships and v2.0-beta users see (or don't see) the popup. If it doesn't work, we fix it; if it does, every future release is in-app from v2.0-beta forward. Linux stays on the current notification-banner approach for now.
- **Maximized window now uses the extra space.** v1.995 unlocked the height; v2.0 lets the content grow into the unlocked space: the Star Log on the Run page gets taller, the Setup tab's scroll area expands. Buttons, banners, and headers stay their normal sizes.
- **One last manual install.** Existing v1.x users download v2.0-beta from the GitHub Releases page one more time. From there, the app starts checking for updates on its own. Whether the auto-install actually delivers the next release cleanly is what this beta is for.
- **Polish:** the rotating astro phrases during the AI warmup no longer cycle back to "Studying your stars" looking like the run restarted; long warmups settle on a steady "Still warming up the AI trail detector" line. The `#StarTrailCleanR` hashtag in the Run-page sharing message is now bold.
- **Smoke tests:** 132 still passing, plus the bundled-app smoke that runs the actual frozen binary on synthetic frames on Mac Apple Silicon, Mac Intel, Windows, and Linux.

## v1.995-beta
- **Fix: maximize now fills the whole screen on Windows.** Warren reported that clicking maximize only filled the top portion of his monitor with empty desktop below. Root cause: yesterday's run-screen layout work locked the window's minimum and maximum height to a single value so Run tab content would match Setup tab perfectly. That same lock prevented Windows from growing the window vertically on maximize. Removed the maximum-height lock; the minimum stays in place so the Setup tab can never clip below the Clean My Stars button. Trade-off: if the window is noticeably taller than the Setup tab's natural height, the Run tab may show empty space at the bottom — a smaller cost than not filling the screen.
- **Fix: displayed output folder uses consistent forward slashes on Windows.** Same Warren screenshot caught a path-separator inconsistency: the auto-output field showed "L:/for Bruce/Camera 1\cleaned" — forward slashes from Qt's file picker, then a backslash where the GUI appended "cleaned". Two conventions in one path. Now consistent forward slashes regardless of how the input folder arrived (browse picker, hand-typed, or pasted from elsewhere).

## v1.994-beta
- **Smaller download.** Roughly 150 MB stripped from the installer through a five-round bundle audit. No functional changes vs v1.993 — trail detection, repair, output formats, run summary, mask editor are all identical. Targets removed: Qt SDK developer tools that ship inside PySide6 (Assistant, Linguist, Designer, qmlls, qmlformat, qmllint, etc. — they're utilities for *building* Qt apps, not running them); Qt's multimedia plugin folder plus the ffmpeg video/audio codec libraries that backed it (we don't process video); the OpenCV video stack, swapped from opencv-python to opencv-python-headless (same OpenCV, no GUI/video bits we never used); and a handful of orphan Python packages the bundler was pulling in transitively (pip, astropy_iers_data, fontTools, on top of the previous round's matplotlib, pandas, lxml, openai, anthropic, imgviz, labelme). UK testers and others on slow connections feel this most.
- **Why now: prep for v2.0.** The v2.0 auto-update story (in-app Sparkle/WinSparkle updates with delta downloads) is the next milestone. Smaller bundle equals smaller deltas equals faster updates for everyone from v2.0 onward, plus a smaller first install for new users in the meantime.

## v1.993-beta
- **Customer experience lens: TIFFs the worker can read should never trigger a "files have a problem" dialog.** v1.992 added a pre-flight modal that surfaces frames being skipped before a run starts. Useful when something is genuinely wrong, but it was firing on TIFFs that Pillow couldn't parse (BigTIFF, unusual compressions, multi-IFD layouts) even though our worker could process them perfectly. The GUI's scan stage now mirrors the worker's reader ladder: tries Pillow first, falls back to tifffile for TIFFs Pillow can't read. Customers with TIFFs from Photoshop's "Save with Layers" mode, scientific-camera converters, or BigTIFF files now have those frames silently process instead of seeing a confusing dialog.
- **Run-summary log gets head+tail trimming for very large runs.** v1.992 appended the full Star Log to the saved summary text file so a single file is enough to diagnose any run question. For typical runs (50-500 frames) the log is 5-50 KB and passes through whole. For very large runs (1000+ frames) the log can grow into hundreds of KB of repetitive per-frame progress lines, which makes the summary hard to scroll. v1.993 keeps the first 50 and last 100 lines (where the diagnostically interesting content clusters — resolution headers, skipped-files notice, errors, completion summary) and elides the repetitive middle with a clear "X lines elided" marker so readers know it was trimmed and roughly how much. Small runs still pass through whole; the trim only kicks in past ~170 lines.
- **Smoke tests:** +1 structural check that the GUI scan keeps the tifffile fallback in place. Total smoke suite: 132 tests.

## v1.992-beta
- **No more silent frame drops.** A Windows tester pointed the app at a folder of 94 TIFFs and only 51 processed; the other 43 vanished without an error message. Root cause: when Star Trail CleanR scans a folder, it picks the dominant image resolution and silently filters out anything that doesn't match. Useful when needed (one bad portrait-orientation frame mixed in), but invisible to the user. v1.992 makes this visible. Before the run starts, if any frames will be skipped — different resolution, unreadable header, anything — a popup shows up listing exactly what's in the folder ("51 frames at 5568×3712, 43 at a different size") with the first few skipped filenames as examples. User can Continue or Cancel and check the folder first.
- **Run summary now records what got skipped.** The plain-text run summary (saved into your input folder's `cleanr_workspace`) now includes a Frames skipped line broken down by reason, so anyone reviewing a run after the fact can see exactly what processed and what didn't, even if they dismissed the popup at the time.
- **Run summary now includes the full Star Log.** Everything that scrolled in the live run window is appended to the bottom of the saved summary text file. One file is now enough to diagnose any run question — no need to ask testers for screenshots of the log scroll. If a future run drops frames or hits a bad file, the saved summary captures the whole story.
- **Cancel-from-modal exits cleanly.** Cancelling the new pre-flight modal aborts the run before any subprocess starts; the GUI surfaces a "Run cancelled" message and returns to the start page. Same plumbing pattern as v1.99's bad-file modal.
- **Smoke tests:** +3 covering the new signal/slot wiring, that the run summary surfaces skipped counts, and that the saved summary appends the full Star Log. Total smoke suite: 131 tests.

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
