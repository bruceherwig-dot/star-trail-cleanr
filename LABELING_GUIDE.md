# Star Trail CleanR — Labeling Guide

Rules for drawing trail polygons in CVAT (or LabelMe). Keep this open while reviewing. Consistent labels train a better detector — that's the whole point.

**Label class:** `trail` (only one class in this project).

---

## The core rules

### 1. Split overlapping trails
Two streaks = two polygons, even if they touch or cross.
Why: the model learns "one polygon = one trail." If you box both together, it learns to draw one big box around any pair of nearby streaks, and you get the giant-green-rectangle problem.

### 2. Tight polygons
Hug the trail edges. Don't leave a generous margin of sky.
Why: loose polygons teach the model that "trail" includes surrounding sky, which hurts segmentation precision during repair.

### 3. Include the full tail
Satellite trails have a bright core and a dimming tail. Follow the tail all the way until it blends into noise — don't stop at the bright section.
Why: the dim tail is the hardest part for the detector to learn, and it's the part most likely to leak through repair if under-labeled.

### 4. Airplane strobes = one trail
A dashed "blink blink blink" pattern is ONE label, not many. Draw one polygon covering the whole dashed run.
Why: each dash is part of the same object's path. Labeling them separately would teach the model that short segments are independent objects.

### 5. Never label star rotation arcs
Curved arcs from Earth's rotation are real stars, NOT trails. Leave them alone.
Why: this is the single biggest false-positive trap. If you label star arcs, the model will start removing real stars during repair.

### 6. Occluded trails — label the visible parts only
Trail passing behind trees/terrain: label each visible segment separately. Don't guess at the hidden path.
Why: the model should only learn what's actually in the pixels. Guessing teaches it to hallucinate.

### 7. Frame edges — label up to the edge
If a trail exits the frame, draw the polygon all the way to the edge. Don't cut it short.
Why: partial trails at edges are still trails. Cutting them short teaches the model to leave stubs behind.

### 8. Never label hot pixels
Single bright dots or small clusters that stay fixed frame-to-frame are hot pixels, not trails. They're handled by a separate pipeline stage.
Why: polluting the trail class with hot pixels teaches the model to classify stars as trails.

---

## Judgment calls

When in doubt:
- **Label it** if it's clearly a moving object (satellite, plane, meteor, debris).
- **Skip it** if it might be a star, reflection, or noise.
- **Skip it** if you can't tell — a missed trail is better than a false trail, because the model can recover from a miss but not from being taught to remove stars.

## Process notes

- **Pre-annotations from the model are a starting point, not truth.** Every pre-annotated polygon needs human review. Split, tighten, or delete as needed.
- **Reviewed JSONs are sacred.** Never regenerate a reviewed batch — CLAUDE.md enforces this.
- **Edge cases go here.** When you hit a case these rules don't cover, add it to this file so future-you (and Claude) apply the same call.

---

## Rule history / additions

Append new rules below as you discover edge cases. Each new rule gets a date and a one-line "why."

- 2026-04-15 — initial ruleset written after Bruce noticed Joshua Tree double-trails during Silvana batch review
