# Star Trail CleanR

**Remove the Trails. Keep the Stars.**

A free desktop app for Mac and Windows that removes airplane and satellite trails from wide-field star trail sequences while preserving the real stars. The result is a clean set of frames you can stack into a perfect star trail composite.

Website: [www.startrailcleanr.com](https://www.startrailcleanr.com)

---

## Download

The links below always point to the latest release.

- [**Mac (Apple Silicon)**](https://github.com/bruceherwig-dot/star-trail-cleanr/releases/latest/download/StarTrailCleanR-Mac-AppleSilicon.zip)
- [**Mac (Intel)**](https://github.com/bruceherwig-dot/star-trail-cleanr/releases/latest/download/StarTrailCleanR-Mac-Intel.zip)
- [**Windows**](https://github.com/bruceherwig-dot/star-trail-cleanr/releases/latest/download/StarTrailCleanRSetup.zip)

See the [Releases page](https://github.com/bruceherwig-dot/star-trail-cleanr/releases) for older versions and full changelogs.

---

## How It Works

Star Trail CleanR runs in two steps:

1. **Trail Detection.** Each frame is run through a YOLO segmentation model trained on thousands of manually labeled airplane and satellite trails across many cameras, lenses, and sky conditions. The model produces pixel-accurate masks for every trail it finds.

2. **The Fix, Star Bridge Repair.** For each trail, Star Trail CleanR pulls clean pixels from the frame immediately before and after, blending them across the trail using a morphing technique called *Star Bridge*. This preserves the real stars underneath the trail and keeps the brightness and color natural. No smudges, no blank patches.

---

## Quick Start

1. **Browse.** Choose your folder of frames.
2. **Mask (optional).** Paint over ground, buildings, and rocks so the AI ignores them. Trees can be left unmasked.
3. **Format.** Pick output format (JPG / TIFF 8-bit / TIFF 16-bit) and JPEG quality.
4. **Run.** Sit back. Cleaned frames land in a `cleaned/` folder next to your originals.
5. **Stack.** Load the cleaned frames into your favorite stacker (StarStaX, Sequator, Photoshop) for the final composite.

---

## Limitations

- **Trail variety is bounded by the AI's training data.** If a type of trail isn't being detected well in your sequences, you can help train the next version: zip 300+ frames from that scene and email them to bruceherwig+startrailcleanr@gmail.com. For large folders, share a Dropbox, Google Drive, or WeTransfer link instead. The model gets smarter every time the community contributes.
- **Meteors will be removed too.** Their streaks look similar to airplane and satellite trails, so the detector cannot tell them apart. If you want to keep them, use your originals to mask them back in.
- **RAW files (.CR2, .NEF, .ARW, etc.) are not yet supported.** Convert your sequence to JPG or TIFF first, then run Star Trail CleanR on the converted frames.
- **Not a one-click fix.** You'll still want to touch up the final composite in Photoshop or your editor of choice. But if we did our job right, it's a fraction of the time you used to spend.
- **Designed for wide-field star trail sequences,** not deep-sky tracked exposures.

---

## About the Authors

Star Trail CleanR is a passion project. I've been shooting star trails for over a decade, and the whole time I kept thinking *somebody should really write a program that gets rid of all the airplane and satellite trails*. Nobody did. So I finally built one, with a lot of help.

After countless hours of back-and-forth with Claude Code, I described what I wanted, Claude wrote the code, we tested it, I pushed back, we tried again. Star Trail CleanR wouldn't exist without that partnership.

Star Trail CleanR is my free gift to the astrophotography community that has taught me so much.

- Project site: [StarTrailCleanR.com](https://startrailcleanr.com)
- Photos for sale: [bruceherwig.com](https://bruceherwigphotographer.square.site/shop/astrophotography/3?page=1&limit=30&sort_by=category_order&sort_order=asc)
- Blog: [bruceherwig.wordpress.com](https://bruceherwig.wordpress.com)

---

## Acknowledgments

Star Trail CleanR exists because of the generosity of fellow astrophotographers who shared their image sequences for AI training, tested early builds, and offered feedback. Thank you, all of you.

See the [full list of contributors](https://bruceherwig.wordpress.com/star-trail-cleanr/#Thanks).

---

## Feedback and Sharing

Got a before-and-after you'd like to share? Got an idea or feedback to make Star Trail CleanR even better? Email [bruceherwig+startrailcleanr@gmail.com](mailto:bruceherwig+startrailcleanr@gmail.com).

---

## Crash Reports

The first time you launch Star Trail CleanR you'll be asked whether you'd like to send anonymous crash reports. If you say yes, the app sends an automatic report (stack trace, operating system, app version) when it crashes, so the bug can be found and fixed. If you say no, nothing is sent. Either way, no images, no folder paths, and no personal information are ever collected.

---

## License

MIT. See [LICENSE](LICENSE).
