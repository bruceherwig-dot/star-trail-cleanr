# Star Trail CleanR

**Remove the Trails. Keep the Stars.**

A free desktop app for Mac and Windows that removes airplane and satellite trails from wide-field star trail sequences while preserving the real stars. The result is a clean set of frames you can stack into a perfect star trail composite.

Website: [www.startrailcleanr.com](https://www.startrailcleanr.com)

---

## Download

The links below always point to the latest release.

**[Download for Mac, Windows, or Linux](https://www.startrailcleanr.com/#download)**

<!-- These go to the website's download buttons ON PURPOSE, rather than to
     GitHub asset URLs. Asset filenames change (Mac moved from .zip to .dmg),
     and hardcoded links here rotted into 404s without anyone noticing. The
     website buttons resolve to the current release for every platform and also
     count the download. Please do not replace them with direct file links. -->

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
5. **Stack.** Build your star trail or timelapse right in the app (the Star Trail & Timelapse window), or load the cleaned frames into your favorite stacker (StarStaX, Sequator, Photoshop, etc.).

---

## What You Get When It Finishes

**Your original photos are never touched.** Everything is written alongside them.

Inside your photo folder you'll find a new **`cleaned`** folder holding one cleaned
copy of every frame, with the same filenames. Inside that is a folder called
**`STC Extras`** with everything else the run made:

| | |
|---|---|
| **STC_cleaned_star_trail.jpg** | your star trail, stacked from the cleaned frames |
| **STC_original_star_trail.jpg** | the same stack from your untouched originals, so you can compare |
| **STC_share_video.mp4** | a short before-and-after video |
| **STC_star_trail_…jpg** / **STC_timelapse_…mp4** | anything you build on the Star Trail or Timelapse tab. The filename records the settings used, so you can tell two versions apart |
| **Star Log** | the one file to open when something looked wrong. If a frame was skipped or a mask was refused, it says so, in plain English |
| **foreground_mask.png** | the mask you painted, reused automatically next time |
| **masks** folder | what the detector found on each frame, if you want to see its work |

**It went well if:** every frame has a cleaned copy, and the star trail opens with
no streaks across the sky.

**Worth a look if:** a streak is still there (the detector missed it), or frames are
missing from the cleaned folder. The Star Log will say why.

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

---

## For developers

`ARCHITECTURE.md` maps the codebase: what runs when a user clicks Clean, which
files matter, and which are kept only for history. Start there rather than with
the source.
