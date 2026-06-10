# How Star Trail CleanR Auto-Update Works (READ THIS FIRST)

This is the single source of truth for how updates reach users. If you — or a
future Claude with no memory of any conversation — need to know how updating
works, read THIS and verify against the cited code lines. **Do not describe
update behavior from memory.** Getting this wrong and stating it confidently is
exactly the mistake this document exists to prevent.

Verified against the actual code on 2026-06-09.

---

## The promise (the agreed v2.0 design)

Every update is seamless and in-app: the user opens the app, gets notified,
clicks once, and the new version downloads, installs **in place**, and restarts
itself. No website, no manual download, no reinstall — **except on Linux**,
which has no built-in installer and uses a download link.

## Updating has two halves — BOTH must work, or users get nothing

### Half 1: PUBLISHING (server side) — putting each new version into the feed

- The "feed" is three XML files (Sparkle/WinSparkle "appcasts"), one per
  platform, hosted on GitHub Pages:
  - `appcast-mac-apple-silicon.xml`, `appcast-mac-intel.xml`, `appcast-windows.xml`
  - at `https://bruceherwig-dot.github.io/star-trail-cleanr/`
- Publishing happens **automatically on every release tag**, in CI:
  `.github/workflows/build.yml` → the **`publish-appcast`** job. It signs each
  installer with the `SPARKLE_ED_PRIVATE_KEY` GitHub secret, prepends the new
  version to each feed (`scripts/publish_appcast.py`), pushes to the `gh-pages`
  branch, then **HARD-FAILS the build if the live feeds don't show the new
  version.**
- **"Release done" = the `publish-appcast` job is GREEN**, not just "the build
  compiled." If that job is red, auto-update users did NOT get the release.
- **History (why this is automated now):** publishing used to be a manual local
  command (`scripts/release_signer.py`) run on Bruce's Mac. It got skipped after
  v2.04-beta. The feed sat **frozen at 2.04 for 39 days** (May 1 – Jun 9, 2026)
  while **43 versions** shipped, so the seamless updater reached nobody on 2.04+.
  That is why publishing is now automatic with a hard-fail verify gate. **Never
  reintroduce a manual "remember to publish" step.**

### Half 2: DELIVERING (app side) — the installed app checking + installing

The installed app has **two independent notifications**. Both run on launch.
Know the difference:

**A) The orange in-app banner (our own custom notice).**
- Code: `star_trail_cleanr.py` → `_start_update_check()` (called every launch in
  the MainWindow setup) → `UpdateCheckThread` → `modules/update_check.py`
  `check_for_update()`; shown by `_on_update_result()`.
- It checks **GitHub's latest release** (NOT the feed) on **every launch**.
- Its "Update" button (`_on_update_download`) runs the **one-click in-place
  installer** on Mac/Windows, and only opens the website on **Linux**.

**B) The built-in one-click installer (Sparkle on Mac, WinSparkle on Windows).**
- Code: `modules/sparkle_updater.py`, `modules/winsparkle_updater.py`; started at
  launch in `star_trail_cleanr.py` (`init_sparkle` / `init_winsparkle`).
- It reads the **appcast feed**. When the feed shows a newer version it pops its
  **own native "A new version is available — Install" window** and does the
  download + in-place install + restart.
- It checks on **every launch** (`check_for_updates_in_background`, added
  v2.48-beta) AND on a daily timer (`SUScheduledCheckInterval = 86400`) as a
  backstop, AND when the user clicks **Settings → Check for Updates**.

## Platform matrix

| Platform | Installer | Feed | Behavior |
|---|---|---|---|
| Mac Apple Silicon | Sparkle | appcast-mac-apple-silicon.xml | one-click in-place |
| Mac Intel | Sparkle | appcast-mac-intel.xml | one-click in-place |
| Windows | WinSparkle | appcast-windows.xml | one-click in-place |
| Linux | none | (banner only) | banner button opens GitHub download page |

## The wrong-location failure (added 2026-06-10, after a real user hit it)

macOS **silently disables Sparkle** when the app runs from the mounted DMG or
from the quarantine sandbox ("App Translocation" — typically when the .app was
not properly dragged into /Applications). Symptoms: the launch splash says
"Checking for updates…" (that text is OURS and appears before Sparkle starts),
no update is ever offered, and Settings → Check for Updates does nothing. A
user on 2.45 (Apple Silicon, latest macOS) hit exactly this and had been
reinstalling from the website every release. Server side was verified healthy
the same day (feed current, signature valid over the shipped DMG, key match,
Sparkle present in the bundle) — the failure was entirely machine-local.

Three defenses (all added 2026-06-10):
1. **DMG layout**: both Mac DMGs now show the app NEXT TO an
   Applications-folder shortcut (`.github/workflows/build.yml`, staged with
   ditto + `ln -s /Applications`), the canonical drag-to-install layout.
2. **Launch guard**: `star_trail_cleanr.py` (after MainWindow shows) detects
   `/AppTranslocation/` or a `/Volumes/…` executable path and tells the user
   to move the app to Applications.
3. **Never-silent button**: `sparkle_updater.check_for_updates()` /
   `winsparkle_updater.check_for_updates()` return False when the engine is
   dead; both button handlers then show `_updater_unavailable_fallback()` (a
   plain-language dialog + the download page) instead of doing nothing.

Diagnostic: every install writes `~/.star_trail_cleanr/sparkle_debug.log` —
it records whether the engine started and what each check attempt did. Ask a
user for that file before theorizing.

## The bootstrapping rule (important, easy to forget)

The updater can only fix itself **going forward**. A user on an OLD version runs
that OLD version's updater code. So a fix to the updater (e.g. the banner button
now doing in-place install instead of opening the website) only takes effect
once the user is on a build that **contains** the fix. That build is delivered
by the existing (already-working) installer; from then on the fixed behavior
applies. A broken updater cannot fix the very update that fixes it.

## How to VERIFY it's actually working (check, never assume)

- **Are the live feeds current?** Fetch each appcast URL and read the top
  `<sparkle:version>`; it must equal the newest non-prerelease release tag.
- **Did the last release publish?** In CI, the `publish-appcast` job must be
  green. Locally, `python3 scripts/publish_appcast.py vX.YY-beta --verify`
  exits non-zero unless all three live feeds advertise that version.
- **Does the app check on launch?** `_start_update_check()` (orange bar) is in
  the MainWindow setup; `check_for_updates_in_background()` is called right after
  `init_sparkle` / `init_winsparkle`.

## Version numbering

Versions are `major.counter` (2.46, 2.47, 2.48 …). They are compared
**component-wise, NOT as floats** (so 2.10 is newer than 2.9, and 2.100 newer
than 2.99). Both `modules/update_check.py` and Sparkle's own comparator do this.
Keep the counter incrementing; write `2.50`, never `2.5`.

## What can only be confirmed on a real build

Sparkle does nothing when running live Python source (it only activates inside
the frozen `.app`), and there is no Windows machine in the dev environment. So
the Mac/Windows in-place popup can only be fully proven by installing a built
release and watching it. The `publish-appcast` verify gate plus the daily
watchdog idea (see todo) are the backstops that catch failures when it can't be
eyeballed.
