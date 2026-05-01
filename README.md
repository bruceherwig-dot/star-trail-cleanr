# Star Trail CleanR auto-update feeds

This branch (`gh-pages`) hosts the appcast XML files that Sparkle (Mac) and WinSparkle (Windows) fetch in the background to discover new releases of [Star Trail CleanR](https://startrailcleanr.com).

## Files

- `appcast-mac-apple-silicon.xml` &#8212; auto-update feed for Mac (Apple Silicon)
- `appcast-mac-intel.xml` &#8212; auto-update feed for Mac (Intel)
- `appcast-windows.xml` &#8212; auto-update feed for Windows

Linux uses a notification-banner approach pointing to GitHub Releases; no appcast feed is published for Linux.

## URLs

The feeds are served at:

- `https://bruceherwig-dot.github.io/star-trail-cleanr/appcast-mac-apple-silicon.xml`
- `https://bruceherwig-dot.github.io/star-trail-cleanr/appcast-mac-intel.xml`
- `https://bruceherwig-dot.github.io/star-trail-cleanr/appcast-windows.xml`

These URLs are baked into the app's bundle (`Info.plist` `SUFeedURL` on Mac, `init_winsparkle()` argument on Windows) and never seen by end users. Sparkle/WinSparkle pop a native update dialog when an item in the feed advertises a version newer than the running app.

## Publish workflow (when a new release ships)

1. Tag the release on `main` (e.g., `v2.0-beta`).
2. CI builds artifacts and uploads them to GitHub Releases.
3. Sign the Mac artifacts with the Sparkle private key (in macOS Keychain): `bin/sign_update <artifact>`.
4. Run `bin/generate_appcast` over the directory of Mac artifacts to produce signed appcast entries.
5. Hand-template the Windows entry (no first-party tool); sign with the same Ed25519 key.
6. Append the new `<item>` blocks to each appcast XML in this branch.
7. Commit and push.

Existing v2.0-beta+ users see the native update popup on their next launch (or within 24 hours if already running).
