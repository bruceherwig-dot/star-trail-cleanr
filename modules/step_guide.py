"""Plain-English guide to every timed step, so a log explains itself.

Every step that records time in modules/step_timer.py has an entry here: a short label, one
sentence saying what the step is for, and what each of its counts means. The run timing block in
the Star Log and the dev run log both print these sentences next to the numbers, so a person (or
an AI) reading a log never needs the code to know what a row is.

THE RULE (Bruce, 2026-11-18): a section of code that is documented in plain English is also timed,
and the plain-English purpose is what its timing row means. tests/test_step_timer.py fails if a step
records itself without an entry here.

Writing style: plain words, no code names, no jargon, no dashes used as punctuation.
"""

# name -> (label, purpose, {count key: what it means})
STEPS = {
    # ---- between the big stages ------------------------------------------------------------
    "batch.raw_labeled_map": (
        "Mapping what the AI first marked",
        "Builds a picture of the AI's raw detections for each frame, kept for the log and for the rescue steps.",
        {}),
    "batch.sky_trim_segments": (
        "Trimming outlines to the sky",
        "Cuts each trail outline back so it never reaches into the foreground, and drops outlines that end up empty.",
        {}),
    "batch.drop_rejected_polygons": (
        "Dropping outlines the static filter erased",
        "Removes from the cleaning and export lists any outline that the static filter mostly erased, so repair does not clean what was rejected.",
        {"outlines_checked": "outlines looked at", "outlines_dropped": "outlines removed",
         "counted_in_own_box": "outlines counted inside their own small box (fast)",
         "counted_whole_frame": "outlines counted across the whole frame (slow, no usable corners)"}),
    "batch.run_log_detect_records": (
        "Writing the detailed detection log",
        "Saves the per-frame detection record to the developer log (only on a developer machine).",
        {}),
    "batch.count_trails": (
        "Counting trails",
        "Counts how many frames and trails this batch found, for the progress line.",
        {}),
    "batch.save_mask_files": (
        "Saving mask files",
        "Writes the optional mask and outline files when they were asked for.",
        {}),
    "batch.count_trails_per_frame": (
        "Counting trails per frame",
        "Counts the trails on each frame just before it is cleaned, for the progress display.",
        {}),
    "batch.read_frame_exif": (
        "Reading each frame's camera data",
        "Reads the capture time, exposure and lens details from the original file so the cleaned frame keeps them.",
        {}),
    "batch.finder_comments": (
        "Stamping the Finder comment (Mac only)",
        "Writes the app name and version into each cleaned file's Finder comment, all files at once.",
        {"stamped_in_one_launch": "files stamped in one go",
         "one_launch_failed_fell_back_to_each_file": "times the one-go stamp failed and files were done one by one",
         "stamped_one_file": "files stamped one by one", "failed_one_file": "files that could not be stamped"}),
    # ---- the stuck-pixel step --------------------------------------------------------------
    "hotpix.search": (
        "Finding stuck pixels",
        "Looks across the batch for pixels that are bright and the wrong colour in almost every frame, which marks a stuck sensor pixel rather than a star.",
        {"colour_first_tiles": "fast-way searches", "whole_numbers_fallback": "slower whole-frame searches",
         "reference_unusual_input": "careful-way searches for unusual frames",
         "frames_searched": "frames looked at", "pixels_marked": "pixels marked as stuck"}),
    "hotpix.map_merge_and_save": (
        "Saving the stuck-pixel map",
        "Adds this batch's findings to the map saved for the whole run.",
        {}),
    "hotpix.ground_check": (
        "Deciding whether to repaint the ground",
        "Checks whether the marked pixels on the ground look like sensor faults or like textured scenery, which must be left alone.",
        {"left_alone_looks_like_scenery": "ground checks that left the ground alone as scenery",
         "ground_repaint_planned": "ground checks that went ahead with repair"}),
    "hotpix.repaint": (
        "Repainting stuck pixels on the ground",
        "Fills each stuck ground pixel from its surroundings in every frame.",
        {"frames": "frames repainted", "ground_pixels": "ground pixels covered"}),
    # ---- phantom pruning -------------------------------------------------------------------
    "phantom.grey_image": (
        "Making a brightness picture",
        "Makes a one-colour brightness copy of the frame used to judge faint detections.",
        {"detections_in": "detections handed to this step"}),
    "phantom.mask_build": (
        "Drawing each detection",
        "Draws every AI detection as its own full-frame shape so they can be compared.",
        {"lean_one_at_a_time": "frames so large they were drawn one at a time to save memory"}),
    "phantom.mask_merge": (
        "Combining all detections into one picture",
        "Merges every detection into a single picture of everything the AI marked.",
        {"boxed": "detections merged using their tile", "whole_frame_fallback": "detections that had to be merged across the whole frame (slower)"}),
    "phantom.sky_estimate": (
        "Estimating the local sky level",
        "Works out how bright the empty sky is around each spot, to tell a faint real streak from nothing.",
        {}),
    "phantom.thin_parts": (
        "Finding thin dotted parts",
        "Finds the thin parts of detections that have no real light under them in the photo.",
        {}),
    "phantom.bridge_and_label": (
        "Joining the dots into lines",
        "Joins nearby dotted parts into candidate lines.",
        {"candidate_lines": "candidate lines found"}),
    "phantom.judge_lines": (
        "Judging each candidate line",
        "Keeps only long enough, dense enough lines as phantoms, meaning the AI traced a line across empty sky.",
        {"removed_as_phantom": "lines removed as phantoms"}),
    "phantom.trim_detections": (
        "Removing phantoms from the detections",
        "Takes the phantom lines out of every detection and drops detections that are almost all phantom.",
        {"dropped": "detections dropped entirely", "trimmed": "detections trimmed"}),
    # ---- polygon fitting -------------------------------------------------------------------
    "fit.mask_build": (
        "Fetching each detection's shape",
        "Gets the shape of each detection (reusing the one already drawn when possible).",
        {}),
    "fit.edge_scan": (
        "Finding each detection's edges",
        "Finds the tightest box around each detection so later steps only look there.",
        {"searched_tile_box": "detections searched inside their tile (fast)", "searched_whole_frame": "detections searched across the whole frame (slow)"}),
    "fit.peel_split": (
        "Cutting crossings into one rectangle per trail",
        "Where several trails cross or lie side by side in one blob, finds the straight lines and cuts the blob into one rectangle per trail.",
        {"blobs_tried": "blobs examined", "peeled_into_pieces": "rectangles made"}),
    "fit.split_crossing": (
        "Splitting simple crossings",
        "The older way of separating two trails that cross, used when the peel step does not apply.",
        {"blobs_tried": "blobs examined", "confirmed_crossings": "real crossings found"}),
    "fit.crossing_evidence": (
        "Checking whether a blob is a tangle",
        "For a blob that would not split, checks whether it still shows two or more line directions, which means a tangle to keep whole.",
        {"kept_whole": "tangles kept as one piece"}),
    "fit.parallel_split": (
        "Separating side-by-side trails",
        "Splits a blob made of two parallel trails stuck together.",
        {"split_into_pieces": "pieces made"}),
    "fit.props": (
        "Measuring each piece",
        "Measures each piece (length, shape, colour) and keeps only the ones that look like a trail.",
        {"kept": "pieces kept as trails", "dropped": "pieces dropped as not trail-like"}),
    "fit.thin_fat": (
        "Thinning fat junctions",
        "Cuts wide junction blobs back to thin bands, or drops them when thin trails already cover them.",
        {"thinned": "blobs cut to thin bands", "dropped_redundant": "blobs dropped because trails already covered them"}),
    "fit.group": (
        "Grouping pieces into trails",
        "Joins nearby pieces that line up into one trail each.",
        {"groups_made": "trails formed"}),
    "fit.poly_fit": (
        "Fitting an outline to each trail",
        "Draws one outline (polygon) around each trail, curved when the trail bends.",
        {}),
    "fit.rescued_bands": (
        "Outlining kept-whole tangles",
        "Paints each kept-whole tangle into the mask, cut into thin bands where possible.",
        {"blobs": "tangles handled"}),
    "fit.finish": (
        "Finishing the outlines",
        "Counts the results and removes duplicate outlines.",
        {}),
    # ---- static filter ---------------------------------------------------------------------
    "static.bright_trail_veto": (
        "Checking if a fixed-looking detection is really a bright trail",
        "Before a detection that repeats in nearby frames is thrown out as a fixed object, checks whether it is much brighter than the sky around it, which means a real trail.",
        {"kept_as_bright": "detections kept as bright trails", "judged_static": "detections judged to be fixed objects"}),
    # ---- repair ----------------------------------------------------------------------------
    "repair.fill_method": (
        "How each trail was filled in",
        "Records which way each trail piece was repaired: from both neighbours, from one, or with plain sky colour when no clean neighbour exists.",
        {}),
    "repair.star_tracking": (
        "Following the stars between frames",
        "Measures how the stars moved between the neighbouring frames so borrowed sky lines up.",
        {"tracked": "trail pieces where the star motion was found", "not_tracked": "trail pieces where it was not found"}),
    "repair.grain_fill": (
        "Painting sky colour with matching grain",
        "Where neither neighbour has clean sky, paints the local sky colour with grain that matches the photo.",
        {"sky_colour_and_grain_painted": "areas painted", "pixels": "pixels painted",
         "no_sky_to_sample_filled_black": "areas filled black because no sky was nearby"}),
    "repair.darken_restore": (
        "Restoring dark foreground under a trail",
        "Where a trail crosses dark foreground, puts the real dark pixels back instead of sky.",
        {"pieces_restored": "trail pieces restored", "pixels_restored": "pixels restored"}),
    # ---- writing frames --------------------------------------------------------------------
    "write.encode": (
        "Saving each cleaned frame",
        "Encodes and saves each cleaned frame in the chosen format.",
        {}),
    "write.finder_comment_queue": (
        "Queuing the Finder comment (Mac only)",
        "Notes which file needs its Finder comment so it can be stamped with the rest of the batch.",
        {"skipped_drive_cannot_hold_it": "files skipped because the drive cannot hold a comment"}),
}

# The big stages shown in the run timing block: (timing key, label, purpose).
STAGES = [
    ("frame_read_s", "Reading frames", "Loads the batch's photos from the drive."),
    ("hot_pixel_s", "Stuck-pixel step", "Finds stuck sensor pixels and repairs the ones on the ground."),
    ("model_load_s", "Loading the AI", "Loads the trail-finding AI model for the batch."),
    ("star_size_s", "Measuring star size", "Measures how big stars look so a star is not mistaken for a trail."),
    ("dp_tiled_inference_s", "AI finding trails", "The AI looks at every piece of every frame for trails."),
    ("dp_prune_phantoms_s", "Removing phantoms", "Removes thin dotted lines the AI traced across empty sky."),
    ("dp_fit_polygons_s", "Outlining trails", "Turns the AI's marks into one clean outline per trail, splitting crossings."),
    ("dp_seam_second_pass_s", "Checking tile seams", "Looks again along the joins between tiles for trails the first pass missed."),
    ("dp_ground_filter_s", "Ground filter", "Drops detections that are really foreground edges."),
    ("filter_small_s", "Dropping tiny pieces", "Removes pieces too small to be a real trail."),
    ("static_fp_s", "Static filter", "Removes detections that sit at the same spot in every frame, which are fixed objects."),
    ("repair_s", "Repairing frames", "Fills each trail with clean sky borrowed from the neighbouring frames."),
    ("write_s", "Writing frames", "Saves each cleaned frame."),
]
