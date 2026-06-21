# How Star Trail CleanR Auto-Update Works (READ THIS FIRST)

This is the single source of truth for how updates reach users. If you — or a
future Claude with no memory of any conversation — need to know how updating
works, read THIS and verify against the cited code lines. **Do not describe
update behavior from memory.** Getting this wrong and stating it confidently is
exactly the mistake this document exists to prevent.

Verified against the actual code on 2026-06-09. **Updated 2026-06-19** for the
banner-primary redesign — the in-app banner is now the primary notification and
the engine no longer pops its own window on launch (see Half 2).

---

## TWO SEPARATE CHANNELS — do not conflate them (this caused real confusion 2026-06-20)

There are TWO independent update channels, and they read DIFFERENT sources:

1. **The orange BANNER** reads **GitHub `/releases/latest`** (the REST API). It only fires when a real
   GitHub **release** exists with a newer version. A higher entry in the appcast feed does NOT make the
   banner appear. (`modules/update_check.py`.)
2. **The Sparkle ENGINE** (the native install window) reads the **appcast XML feed** on gh-pages. The
   banner's Download button and Settings → Check for Updates both drive this engine, which then shows
   its own install window. (`modules/sparkle_updater.py` + the appcasts.)

So to TEST the banner you need a real release; to test the Sparkle install window you need an appcast
entry. They are not interchangeable. (2026-06-20: a test was set against the appcast while expecting the
banner to fire; it did not, because the banner reads releases, not the appcast.)

## VERIFY ON A REAL FROZEN BUILD — the only proof that counts (rule added 2026-06-20)

Three separate updater bugs shipped because the release was called "done" on a proxy (it built, the
engine loaded, tests passed) that never touched the real update path: the dead Sparkle engine (2.51),
the SSL banner failure (certifi, 2.58), and the BOOL-arg delegate crash (2.58). All three live in the
Python ↔ Sparkle ↔ macOS boundary, which dev and CI cannot see.

**RULE: no release is shipped until a real frozen build has been watched doing a full update end to end
(banner appears → click → install window opens) WITHOUT crashing.** The repeatable test:
1. Publish a throwaway `vX.YY-test` GitHub release (higher number; cancel the CI it triggers) so the
   banner fires, and add a matching temp top entry to the appcast feed so the engine has something to show.
2. Install the real build, relaunch, confirm the banner appears, click it, confirm the Sparkle install
   window opens with no crash. Do NOT install. Click Skip.
3. Delete the throwaway release + tag and revert the appcast. (`/releases/latest` may 404 for a few
   seconds after deleting the latest release; it recomputes.)

## Two frozen-app fragilities that MUST stay fixed (2026-06-20)

- **SSL/certs:** the banner + model-update checks (`update_check.py`, `model_update.py`) verify against
  `certifi.where()`, and `build_helper.py` does `--collect-all certifi`. The frozen app cannot rely on
  the system root store. If certifi is dropped, the checks SSL-fail silently. Guarded by
  `tests/test_update_ssl.py`.
- **PyObjC delegate signatures:** an Objective-C delegate method with a non-object arg (e.g. a BOOL)
  MUST carry an explicit `objc.selector` signature, or PyObjC reads the value as a pointer and SIGSEGVs.
  This is why `standardUserDriverWillHandleShowingUpdate:forUpdate:state:` was REMOVED rather than kept
  (it took a BOOL and crashed the whole app when Sparkle showed an update). See the comment in
  `modules/sparkle_updater.py` and the guard in `tests/test_update_ssl.py`.

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

**Redesigned 2026-06-19 (the "banner-primary" model).** There is now ONE primary,
always-visible notification — the in-app amber banner — plus the one-click install
engine behind it. The engine **no longer pops its own window on launch.**

Why the change: on the 2.51 build (2026-06-19) the engine's native popup opened
**behind** the main window, where the user never saw it — and the banner was being
*suppressed* because the engine was alive. Both channels went silent at once and
the app looked like its updater was dead. So the banner (which lives inside the
window and cannot hide) became primary, and the engine's unprompted launch popup
was removed.

**A) The amber in-app banner — the PRIMARY notification (every platform).**
- Code: `star_trail_cleanr.py` → `_start_update_check()` (every launch in the
  MainWindow setup) → `UpdateCheckThread` → `modules/update_check.py`
  `check_for_update()`; shown by `_reveal_update_banner()`, called from either
  `_on_update_result()` (live check) or `_show_cached_update_banner()` (sticky).
- It checks **GitHub's latest release** (the REST API `releases/latest`, NOT the
  appcast feed) on **every launch**. If that tag is newer than the running
  version, the banner shows. It lives inside the main window, so unlike a
  free-floating popup it **cannot hide behind the window.**
- It shows **whenever a newer release exists** — it is NOT suppressed by the
  engine being alive any more. (That suppression was removed 2026-06-19: with the
  engine no longer auto-popping, suppressing the banner left users with NO visible
  notice.) Only a per-release **dismiss** (the X button → `dismissed_update_tag`)
  keeps it hidden.
- **Resilience (added 2026-06-19 — the check is online and can be slow):**
  - The background check **retries with a generous timeout**
    (`check_for_update(VERSION, timeout_s=12, retries=3)`). It runs off the UI
    thread, so it never delays startup; one slow GitHub response no longer blanks
    the banner. (The check routinely takes ~3.5s — a single slow moment was
    tipping it past the old 5s budget.)
  - **Sticky memory**: a found tag is persisted (`last_seen_update_tag`). On the
    next launch the banner shows **instantly from memory** if that tag is still
    newer than the running version and not dismissed — BEFORE/without a successful
    live check — so a transient timeout or a brief offline launch can't make a
    known update silently disappear. The live check then confirms/refreshes it.
  - Guarded by `tests/test_update_banner.py` (in `tests/run_all.py`): it fails the
    build if the banner stops showing, **including a guard that simulates
    engine-alive + frozen** — the exact condition a re-added suppression would
    hide.
- Its **Update/Download button** (`_on_update_download`) drives the one-click
  in-place installer on Mac/Windows (engine B), shows the explain-and-open-website
  fallback when the engine is dead, and opens the website on **Linux**.

**B) The one-click install engine (Sparkle on Mac, WinSparkle on Windows).**
- Code: `modules/sparkle_updater.py`, `modules/winsparkle_updater.py`; started at
  launch in `star_trail_cleanr.py` (`init_sparkle` / `init_winsparkle`).
- It reads the **appcast feed** and does the download + in-place install + restart.
- It is **NO LONGER triggered automatically on launch** — the
  `check_for_updates_in_background()` launch call was removed 2026-06-19. It is
  driven by the **banner's Update button** and **Settings → Check for Updates**.
  The engine still STARTS on launch (so those buttons work and the dead-engine
  fallback can tell when it didn't); it just doesn't throw up its own window
  unprompted.
- When the engine DOES show its window (from the banner button or Settings), the
  app pulls itself to the front so that window can't open behind the main window
  (`bring_app_to_front()` + the `SPUStandardUserDriverDelegate`
  "willHandleShowingUpdate" hook in `sparkle_updater.py`).
- Sparkle's own daily timer (`SUScheduledCheckInterval = 86400`) still runs as the
  engine's internal backstop.

## Platform matrix

| Platform | Installer | Feed | Behavior |
|---|---|---|---|
| Mac Apple Silicon | Sparkle | appcast-mac-apple-silicon.xml | one-click in-place |
| Mac Intel | Sparkle | appcast-mac-intel.xml | one-click in-place |
| Windows | WinSparkle | appcast-windows.xml | one-click in-place |
| Linux | none | (banner only) | banner button opens GitHub download page |

## Model updates — a SEPARATE channel from app updates (verified against code 2026-06-17)

The AI model (the trail detector, `best.pt`) updates on its OWN path, independent of the
app/appcast above. A new model does NOT need a new app version to reach users.

- Code: `modules/model_update.py` `check_for_model_update()`; GUI shows it via
  `_start_model_update_check()` → `ModelUpdateCheckThread` → `_on_model_update_result()`
  (the orange **model-update card**, separate from the app banner).
- On **every launch**, it queries **all** GitHub releases (`/releases?per_page=100`, so it
  sees prereleases), finds the newest tag matching **`model-v<number>`** (e.g. `model-v5`),
  and compares it to the model in use. "In use" = a model the user previously downloaded
  into their app folder (`get_installed_model_version()`) if present, else the bundled
  `BUNDLED_MODEL_VERSION` (currently **`model-v4`**, in `modules/model_update.py`).
- If a release is **strictly newer** AND has a **`.pt` asset attached**, the card appears
  with the release's first-line summary + a `Credits:` line. One click downloads the `.pt`
  into the user's app folder (`save_installed_model_version`), and that downloaded model
  then takes precedence over the bundled one. It is an **opt-in download prompt, not a
  silent push.**

### To SHIP a new model (e.g. v5)
1. Create a GitHub release tagged **exactly `model-v5`** (the matcher needs the
   `model-v<number>` form; `model-v5.1` is fine, `v5`/`trail-v5` are NOT detected).
2. **Attach `best.pt`** (the first `.pt` asset is what downloads).
3. **Mark it PRERELEASE** — a full release would make `/releases/latest` resolve to the
   model tag and break the website's permanent app-download link (see CLAUDE.md).
4. Optional: put a one-line summary + a `Credits: <name>` line in the release notes — both
   show on the card.
- Publishing is **MANUAL** (`gh release create model-v5 --prerelease` with `best.pt`).
  There is **no CI automation for `model-v*` tags** (unlike app `v*` tags, which auto-publish
  the appcast). Confirmed: nothing in `.github/workflows/` reacts to `model-v*`.
- The bundled `BUNDLED_MODEL_VERSION` only changes when a future APP build bakes in a newer
  `best.pt` and bumps that constant + `_DEV_FALLBACK_MODEL` (CLAUDE.md rule). The
  model-release path reaches existing users WITHOUT that app build.

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
- **Does the app notify on launch?** `_start_update_check()` (the amber banner)
  runs in the MainWindow setup and shows the banner from the live GitHub check or
  from sticky memory. The engine (`init_sparkle` / `init_winsparkle`) still starts
  on launch but no longer auto-pops — `check_for_updates_in_background()` is NOT
  called on launch any more (removed 2026-06-19).
- **Does the banner survive a flaky network?** `tests/run_all.py` →
  `test_update_banner.py` must be green (it locks the banner-shows behavior,
  including the engine-alive-and-frozen no-suppression guard).

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
