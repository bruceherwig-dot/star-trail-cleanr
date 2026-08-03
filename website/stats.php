<?php
/*
 * Star Trail CleanR community stats.
 * Reads the anonymous usage log and returns aggregate totals + breakdowns as
 * JSON, for both the homepage counter and the full /stats.html page. Our own
 * dev/test runs (dev=true) are excluded so published numbers are real. No
 * per-user or per-image data is ever exposed -- counts only. Open-ended lists
 * (cameras, lenses, focal lengths, countries) are ranked by number of unique
 * photographers (install IDs) so one heavy user can't skew them.
 *
 * WHAT COUNTS WHERE (keep this in step with the notes on stats.html):
 *  - Test batches (20 frames or fewer) get NO vote in any breakdown below the
 *    frame gate: source files, orientation, GPU vs CPU, gear, exposure settings,
 *    the full recipe, the no-EXIF share, and the average-run facts. The app
 *    tells people to run a short batch first, so counting those as real sets
 *    skews every list.
 *  - ONE VOTE PER PHOTOGRAPHER: cameras, brands, lenses, focal lengths, ISO,
 *    shutter, aperture, and GPU vs CPU. One heavy user cannot skew these.
 *  - ONE VOTE PER RUN (a heavy user CAN move these, by design): source files,
 *    orientation, and the full recipe, which is meant to show every set.
 *  - Photographer-level facts count from ANY run, test batch included: the
 *    photographer total, country, platform, version, and GPU-owner share.
 *    Filtering those would delete real people who only ever ran a warm-up.
 *  - Headline trails/hours count every run, test batches included -- those
 *    trails were really removed.
 */
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Cache-Control: public, max-age=300');

$REPORTS  = '/home/dh_bmigjp/stc_data/reports.jsonl';

// Community-impact seed (estimated usage across everyone who's downloaded,
// set 2026-06-25). Measured opted-in counts climb on top of it.
$BASELINE_TRAILS = 224003;
$BASELINE_HOURS  = 1872;
// Downloads = a seed for everyone we can't see, plus the count of real people we
// CAN see (opted-in install IDs). The GitHub download feed was dropped: it counts
// every installer download across all versions (updates + re-downloads), which
// wildly overcounts individuals. Seed chosen so the live total reads 354 today;
// it climbs by one per newly identified person.
$BASELINE_DOWNLOADS = 340;

function add_user(&$map, $key, $id) {
    if ($key === '' || $key === null || $id === '') return;
    if (!isset($map[$key])) $map[$key] = array();
    $map[$key][$id] = true;
}

function ranked($map, $tiebreak = 'alpha') {
    // Rank by unique-user count (descending). Break ties so the order is stable
    // and sensible: 'numeric' for focal lengths ("14 mm" -> 14, ascending, so 6
    // sorts before 10), 'alpha' (default) for names (cameras, lenses, countries).
    $out = array();
    foreach ($map as $k => $set) $out[] = array('name' => $k, 'count' => count($set));
    usort($out, function ($a, $b) use ($tiebreak) {
        if ($a['count'] !== $b['count']) return $b['count'] - $a['count'];
        if ($tiebreak === 'numeric') return ((int) $a['name']) - ((int) $b['name']);
        return strcasecmp($a['name'], $b['name']);
    });
    return $out;
}

function clean_camera($s) {
    $s = preg_replace('/\b(corporation|company|imaging|optical|co\.?|ltd\.?|inc\.?)\b/i', '', $s);
    $s = preg_replace('/\s+/', ' ', trim($s));
    $words = explode(' ', $s);
    $out = array(); $prev = '';
    foreach ($words as $w) {
        if ($w !== '' && strcasecmp($w, $prev) !== 0) { $out[] = $w; $prev = $w; }
    }
    if ($out) $out[0] = ucfirst(strtolower($out[0]));   // tidy the brand, keep model codes as-is
    return implode(' ', $out);
}

// Best-effort sensor size from the camera model name. Used to jumpstart the
// full-frame-vs-crop split from the cameras we already collect, before the more
// exact crop-factor method (focal_35mm / focal_length) has data. An unfamiliar
// model returns '' so it lands in "Not determined" rather than being guessed.
// Checks are ordered so the APS-C exceptions win before the broad full-frame
// patterns (e.g. Nikon Z fc / Z50 before the Z full-frame bodies).
function sensor_class($cam) {
    $c = preg_replace('/[\s_\-]+/', '', strtolower($cam));   // "Nikon Z 7_2" -> "nikonz72", "ILCE-7M3" -> "ilce7m3"
    // Medium format
    if (strpos($c, 'gfx') !== false || strpos($c, '645') !== false) return 'Medium format';
    // Micro Four Thirds (Olympus / OM / Panasonic Lumix)
    if (strpos($c, 'olympus') !== false || strpos($c, 'omdigital') !== false || strpos($c, 'omsystem') !== false
        || strpos($c, 'lumix') !== false || preg_match('/panasonicdc|dcgh|emmark|em1|em5|em10/', $c)) return 'Micro Four Thirds';
    // APS-C exceptions that would otherwise match a full-frame pattern -- check first
    if (preg_match('/z(fc|50|30)/', $c)) return 'APS-C';                 // Nikon DX mirrorless
    if (preg_match('/r(7|10|50|100)(\b|$)/', $c)) return 'APS-C';        // Canon APS-C RF
    if (preg_match('/(7d|90d|80d|70d|60d|50d|eosm)/', $c)) return 'APS-C'; // Canon xxD / EOS M
    if (preg_match('/rebel|kiss/', $c) || preg_match('/eos\d{3,4}d/', $c)) return 'APS-C'; // Canon Rebel / xxxD
    if (strpos($c, 'fujifilm') !== false || strpos($c, 'fujix') !== false || preg_match('/[^a-z]x[a-z]?\d/', $c)) {
        return (strpos($c, 'gfx') !== false) ? 'Medium format' : 'APS-C';   // Fujifilm X = APS-C
    }
    if (preg_match('/ilce6|a6\d00|zve10/', $c)) return 'APS-C';          // Sony APS-C
    if (preg_match('/d(500|7\d00|5\d00|3\d00)/', $c)) return 'APS-C';    // Nikon DX DSLR
    // Full frame
    if (preg_match('/z(9|8|7|6|5)/', $c) || strpos($c, 'zf') !== false) return 'Full frame';   // Nikon Z full-frame + Zf
    if (preg_match('/d(6|5|4|850|810|800|780|750|700|610|600)/', $c) || strpos($c, 'df') !== false) return 'Full frame'; // Nikon FX DSLR
    if (strpos($c, 'eosr') !== false && preg_match('/eosr(3|5|6|8|p)?(\b|$|[^0-9])/', $c)) return 'Full frame'; // Canon RF full-frame
    if (preg_match('/1dx|5d|6d/', $c)) return 'Full frame';             // Canon EF full-frame
    if (preg_match('/ilce(7|9|1)|[^a-z]a(7|9|1)(\b|[^0-9])/', $c)) return 'Full frame';  // Sony full-frame
    if (preg_match('/k1(\b|[^0-9])/', $c)) return 'Full frame';         // Pentax K-1
    return '';
}

function country_name($code) {
    // Turn a 2-letter ISO country code into a full name for the stats page.
    // Uses PHP intl for complete coverage when it's available; otherwise a
    // common-country table; and falls back to the raw code for anything unmapped.
    $code = strtoupper(trim((string) $code));
    if ($code === '') return $code;
    if (class_exists('Locale')) {
        $n = @Locale::getDisplayRegion('-' . $code, 'en');
        if (is_string($n) && $n !== '' && strtoupper($n) !== $code) return $n;
    }
    static $N = array(
        'US'=>'United States','CA'=>'Canada','MX'=>'Mexico','GB'=>'United Kingdom','IE'=>'Ireland',
        'FR'=>'France','DE'=>'Germany','AT'=>'Austria','CH'=>'Switzerland','NL'=>'Netherlands',
        'BE'=>'Belgium','LU'=>'Luxembourg','ES'=>'Spain','PT'=>'Portugal','IT'=>'Italy',
        'GR'=>'Greece','SE'=>'Sweden','NO'=>'Norway','DK'=>'Denmark','FI'=>'Finland',
        'IS'=>'Iceland','PL'=>'Poland','CZ'=>'Czechia','SK'=>'Slovakia','HU'=>'Hungary',
        'RO'=>'Romania','BG'=>'Bulgaria','HR'=>'Croatia','SI'=>'Slovenia','RS'=>'Serbia',
        'UA'=>'Ukraine','RU'=>'Russia','EE'=>'Estonia','LV'=>'Latvia','LT'=>'Lithuania',
        'TR'=>'Turkey','IL'=>'Israel','AE'=>'United Arab Emirates','SA'=>'Saudi Arabia',
        'IN'=>'India','PK'=>'Pakistan','CN'=>'China','HK'=>'Hong Kong','TW'=>'Taiwan',
        'JP'=>'Japan','KR'=>'South Korea','TH'=>'Thailand','VN'=>'Vietnam','PH'=>'Philippines',
        'MY'=>'Malaysia','SG'=>'Singapore','ID'=>'Indonesia','AU'=>'Australia','NZ'=>'New Zealand',
        'BR'=>'Brazil','AR'=>'Argentina','CL'=>'Chile','CO'=>'Colombia','PE'=>'Peru',
        'ZA'=>'South Africa','EG'=>'Egypt','MA'=>'Morocco','NG'=>'Nigeria','KE'=>'Kenya',
    );
    return isset($N[$code]) ? $N[$code] : $code;
}

function format_label($k) {
    $lab = array('jpg'=>'JPEG','jpeg'=>'JPEG','tif'=>'TIFF','tiff'=>'TIFF','raw'=>'RAW','png'=>'PNG');
    return isset($lab[$k]) ? $lab[$k] : strtoupper($k);
}

function shutter_label($v) {
    // Exposure time -> a readable label. Star-trail exposures are whole seconds
    // ("30s"); anything under a second becomes a fraction ("1/200").
    $v = (float) $v;
    if ($v <= 0) return '';
    if ($v >= 1) return round($v) . 's';
    return '1/' . round(1 / $v);
}

function aperture_label($v) {
    // f-number -> "f/2.8", dropping any trailing .0 ("f/2", not "f/2.0").
    return 'f/' . rtrim(rtrim(sprintf('%.1f', (float) $v), '0'), '.');
}

function platform_label($platform, $arch) {
    // Map the telemetry platform + arch to a friendly OS bucket. Apple Silicon
    // (arm64) vs Intel Mac (x86_64) matters because Silicon has the GPU built in
    // and Intel does not; Windows GPU is a separate CUDA download.
    $p = strtolower((string) $platform); $a = strtolower((string) $arch);
    if ($p === 'darwin')  return ($a === 'arm64') ? 'Apple Silicon' : 'Intel Mac';
    if ($p === 'windows') return 'Windows';
    if ($p === 'linux')   return 'Linux';
    return $p !== '' ? ucfirst($p) : '';
}

$trails = 0; $runs = 0; $timelapses = 0; $startrails = 0; $no_exif = 0;
$real_runs = 0; $real_frames = 0; $real_trails = 0; $real_gpu = 0;  // runs over 20 frames only
$users = array();
$gpu_users = array();   // photographers with >=1 GPU run (for the Windows-GPU tile)
$fmt = array();
$orient = array('Landscape' => 0, 'Portrait' => 0);   // per-run framing, by width vs height
$gpu_cpu = array();   // compute device, ONE VOTE PER PHOTOGRAPHER (not per run)
$cam = array(); $brand = array(); $lens = array(); $focal = array(); $country = array();
$cam_users = array();   // photographers who reported a camera name on at least one real run
$sensor = array();      // full frame / APS-C / etc., ONE VOTE PER PHOTOGRAPHER
$sensor_users = array();// photographers we could classify by sensor size at least once
$iso = array(); $shutter = array(); $aperture = array();   // EXIF exposure settings
$recipe = array();  // full recipe (ISO, shutter, aperture, focal mm) together, as shot -- PER RUN
$mp = array();      // megapixels per set, whole-MP buckets -- PER RUN, like the recipe
$user_plat = array();   // install_id -> OS bucket (Apple Silicon / Intel Mac / Windows)
$user_version = array(); // install_id -> latest app_version reported (adoption)

if (is_readable($REPORTS)) {
    $fh = fopen($REPORTS, 'r');
    if ($fh) {
        while (($line = fgets($fh)) !== false) {
            $line = trim($line);
            if ($line === '') continue;
            $rec = json_decode($line, true);
            if (!is_array($rec) || !isset($rec['report']) || !is_array($rec['report'])) continue;
            $r = $rec['report'];
            if (isset($r['dev']) && $r['dev'] === true) continue;
            $id = isset($r['install_id']) ? $r['install_id'] : '';
            if ($id !== '') $users[$id] = true;
            if ($id !== '' && !empty($r['platform'])) $user_plat[$id] = platform_label($r['platform'], isset($r['platform_arch']) ? $r['platform_arch'] : '');
            $ctry = isset($rec['country']) ? $rec['country'] : '';
            $type = isset($r['type']) ? $r['type'] : 'run';
            // Version adoption: every report (run OR timelapse) carries app_version.
            // Reports are appended in time order, so the last one wins = current build.
            if ($id !== '' && !empty($r['app_version'])) $user_version[$id] = (string) $r['app_version'];
            if ($type === 'timelapse') { $timelapses++; continue; }
            // Deliberate Star Trail tab renders (v2.83+). The automatic trail built
            // during every run is not reported, so this counts real button presses.
            if ($type === 'startrail') { $startrails++; continue; }
            $runs++;
            // A photographer "has a GPU" if any of their runs (any length) used it.
            if ($id !== '' && isset($r['gpu']) && $r['gpu'] === true) $gpu_users[$id] = true;
            // Country is a per-PHOTOGRAPHER fact, so it counts from ANY run,
            // test batch or not. Filtering it would delete real people from the
            // map just because their only run was a warm-up batch.
            if ($ctry !== '') add_user($country, country_name($ctry), $id);
            $tr = isset($r['trails']) ? (int) $r['trails'] : 0;
            $fr = isset($r['frames']) ? (int) $r['frames'] : 0;
            $trails += $tr;
            // Short test batches (20 frames or fewer -- the app tells people to
            // run a short batch before the full job) would skew the typical-run
            // facts, so exclude them from those averages. Trails still count in
            // the headline total above.
            if ($fr > 20) {
                $real_runs++;
                $real_frames += $fr;
                $real_trails += $tr;
                if (isset($r['gpu']) && $r['gpu'] === true) $real_gpu++;
            }
            // EVERYTHING BELOW is a per-RUN breakdown (source files, orientation,
            // GPU vs CPU, gear, and the exposure settings), so a test batch does
            // not get a vote -- counting a 20-frame warm-up as a real set skews
            // every list under it. The photographer-level facts above (country,
            // platform, version, GPU-owner, headline trails) already counted this
            // report, so nobody disappears from those.
            if ($fr <= 20) continue;
            $f = isset($r['input_format']) ? strtolower($r['input_format']) : '';
            if ($f !== '') { $L = format_label($f); $fmt[$L] = (isset($fmt[$L]) ? $fmt[$L] : 0) + 1; }
            // Orientation: one vote per run. Wider than tall = landscape, taller
            // than wide = portrait; square or missing dims are skipped.
            $w = isset($r['width'])  ? (int) $r['width']  : 0;
            $h = isset($r['height']) ? (int) $r['height'] : 0;
            if ($w > 0 && $h > 0) {
                if ($w > $h)      $orient['Landscape']++;
                elseif ($h > $w)  $orient['Portrait']++;
            }
            // Megapixels per set: one vote per run, like the recipe -- the size
            // is a property of the set. Whole-MP buckets ("24 MP"); prefers the
            // reported figure, falls back to width x height.
            $mpv = isset($r['megapixels']) ? (float) $r['megapixels']
                 : (($w > 0 && $h > 0) ? $w * $h / 1000000.0 : 0);
            if ($mpv > 0) {
                $mk = ((int) round($mpv)) . ' MP';
                $mp[$mk] = (isset($mp[$mk]) ? $mp[$mk] : 0) + 1;
            }
            // GPU vs CPU: ONE VOTE PER PHOTOGRAPHER, same as the gear lists, so
            // someone who runs ten times on a GPU can't skew the split. A person
            // who has used both devices is counted under both.
            if (isset($r['gpu'])) add_user($gpu_cpu, $r['gpu'] === true ? 'GPU' : 'CPU', $id);
            if (empty($r['camera'])) $no_exif++;
            else {
                if ($id !== '') $cam_users[$id] = true;   // this photographer reported a camera at least once
                $cc = clean_camera($r['camera']);
                add_user($cam, $cc, $id);
                // Brand = the first word of the tidied camera name (Canon, Nikon,
                // Fujifilm...). One photographer can appear under more than one
                // brand if they've run different makes -- same as the cameras list.
                $bn = explode(' ', $cc)[0];
                if ($bn !== '') add_user($brand, $bn, $id);
            }
            // Sensor size: full frame vs crop. Prefer the exact crop factor
            // (focal_35mm / focal_length) when the report carries it; otherwise
            // fall back to the camera-model lookup so we get a split from existing
            // data. One vote per photographer, like the other gear lists.
            $scls = '';
            if (!empty($r['focal_length']) && !empty($r['focal_35mm']) && (float) $r['focal_length'] > 0) {
                $ratio = (float) $r['focal_35mm'] / (float) $r['focal_length'];
                if ($ratio < 0.9)       $scls = 'Medium format';   // 35mm-equiv is wider than actual
                elseif ($ratio < 1.2)   $scls = 'Full frame';
                elseif ($ratio < 1.75)  $scls = 'APS-C';
                elseif ($ratio < 2.3)   $scls = 'Micro Four Thirds';
                else                    $scls = '';                // smaller than 4/3 -- leave undetermined
            } elseif (!empty($r['camera'])) {
                $scls = sensor_class($r['camera']);
            }
            if ($scls !== '') {
                add_user($sensor, $scls, $id);
                if ($id !== '') $sensor_users[$id] = true;
            }
            if (!empty($r['lens'])) {
                $ln = trim($r['lens']);
                if ($ln !== '' && !preg_match('/^0+(\.0+)?\s*mm/i', $ln) && stripos($ln, 'f/0') === false) add_user($lens, $ln, $id);
            }
            if (isset($r['focal_length'])) add_user($focal, ((int) round($r['focal_length'])) . ' mm', $id);
            if (!empty($r['iso']))          add_user($iso, (string) (int) $r['iso'], $id);
            if (!empty($r['exposure_sec']) && shutter_label($r['exposure_sec']) !== '') add_user($shutter, shutter_label($r['exposure_sec']), $id);
            if (!empty($r['aperture']))     add_user($aperture, aperture_label($r['aperture']), $id);
            // The full recipe, exactly as the set was shot: ISO, shutter, f-stop,
            // and focal length together. Counted PER RUN (every set votes).
            // Aperture and focal length both come from the LENS, so when both are
            // missing we collapse them to a single "lens not reporting". A set that
            // reported NONE of the four (no usable EXIF at all) goes into a single
            // "No data collected" bucket, pinned to the bottom of the list below.
            $has_iso = !empty($r['iso']);
            $has_sh  = !empty($r['exposure_sec']) && shutter_label($r['exposure_sec']) !== '';
            $has_ap  = !empty($r['aperture']);
            $has_fl  = isset($r['focal_length']);
            if (!$has_iso && !$has_sh && !$has_ap && !$has_fl) {
                $recipe['No data collected'] = (isset($recipe['No data collected']) ? $recipe['No data collected'] : 0) + 1;
            } else {
                $iso_s = $has_iso ? 'ISO ' . (int) $r['iso'] : 'ISO not reported';
                $sh_s  = $has_sh ? shutter_label($r['exposure_sec']) : 'not reported';
                if (!$has_ap && !$has_fl) {
                    $lens_s = 'lens not reporting';
                } else {
                    $ap_s = $has_ap ? aperture_label($r['aperture']) : 'not reported';
                    $fl_s = $has_fl ? ((int) round($r['focal_length'])) . ' mm' : 'not reported';
                    $lens_s = $ap_s . " \u{00B7} " . $fl_s;
                }
                $rk = $iso_s . " \u{00B7} " . $sh_s . " \u{00B7} " . $lens_s;
                $recipe[$rk] = (isset($recipe[$rk]) ? $recipe[$rk] : 0) + 1;
            }
        }
        fclose($fh);
    }
}

$fmt_list = array();
foreach ($fmt as $k => $n) $fmt_list[] = array('name' => $k, 'count' => $n);
usort($fmt_list, function ($a, $b) { return $b['count'] - $a['count']; });

// Megapixels: most-common first; ties break small-to-large so the list is stable.
$mp_list = array();
foreach ($mp as $k => $n) $mp_list[] = array('name' => $k, 'count' => $n);
usort($mp_list, function ($a, $b) {
    if ($a['count'] !== $b['count']) return $b['count'] - $a['count'];
    return ((int) $a['name']) - ((int) $b['name']);
});

// Full recipe (ISO, shutter, aperture, focal mm), ranked by how many sets were
// shot that way, most-used first.
$recipe_list = array();
foreach ($recipe as $k => $n) $recipe_list[] = array('name' => $k, 'count' => $n);
usort($recipe_list, function ($a, $b) {
    // Primary: most-used first. Secondary: ISO ascending (numeric, so "ISO 800"
    // sorts before "ISO 1000"). Then the rest of the text for a stable order.
    if ($a['count'] !== $b['count']) return $b['count'] - $a['count'];
    $ai = preg_match('/ISO (\d+)/', $a['name'], $m) ? (int) $m[1] : PHP_INT_MAX;
    $bi = preg_match('/ISO (\d+)/', $b['name'], $n) ? (int) $n[1] : PHP_INT_MAX;
    if ($ai !== $bi) return $ai - $bi;
    return strcasecmp($a['name'], $b['name']);
});
// Sink the incomplete rows to the bottom regardless of count: complete recipes
// first (ranked), then the "lens not reporting" sets grouped together, then the
// "No data collected" bucket last -- same idea as the country/brand Unknown rows.
$no_data = null; $complete = array(); $lens_missing = array();
foreach ($recipe_list as $row) {
    if ($row['name'] === 'No data collected') $no_data = $row;
    elseif (strpos($row['name'], 'lens not reporting') !== false) $lens_missing[] = $row;
    else $complete[] = $row;
}
$recipe_list = array_merge($complete, $lens_missing);
if ($no_data !== null) $recipe_list[] = $no_data;

$orient_list = array();
foreach ($orient as $k => $n) if ($n > 0) $orient_list[] = array('name' => $k, 'count' => $n);
usort($orient_list, function ($a, $b) { return $b['count'] - $a['count']; });

// Counts PHOTOGRAPHERS per device now, so ranked() (which counts the unique-ID
// set behind each key) is the right reader, exactly like the gear lists.
$gpu_cpu_list = ranked($gpu_cpu);

// Photographers whose country couldn't be determined (reported before GeoIP went
// live, or the lookup failed; the IP is discarded so it can't be backfilled).
// Shown as an "Unknown" row pinned to the bottom so the country breakdown
// reconciles with the photographer count, but NOT counted as a country.
$located = array();
foreach ($country as $set) { foreach ($set as $uid => $_) $located[$uid] = true; }
$unknown_users = 0;
foreach ($users as $uid => $_) { if (!isset($located[$uid])) $unknown_users++; }
$countries_list = ranked($country);
if ($unknown_users > 0) $countries_list[] = array('name' => 'Unknown', 'count' => $unknown_users);

// Camera brands: append an "Unknown" row for photographers who never reported a
// readable camera name (RAW files and stripped JPEGs don't carry one), pinned to
// the bottom so the brand percentages reconcile with the full photographer count
// instead of only the ones we could identify -- same idea as the country Unknown.
$brand_unknown = count($users) - count($cam_users);
$brands_list = ranked($brand);
if ($brand_unknown > 0) $brands_list[] = array('name' => 'Unknown', 'count' => $brand_unknown);

// Full frame vs crop: ranked by photographers, with a "Not determined" row for
// people we couldn't classify (no camera name and no crop factor), pinned to the
// bottom so the split reconciles with the full photographer count.
$sensor_undet = count($users) - count($sensor_users);
$sensor_list = ranked($sensor);
if ($sensor_undet > 0) $sensor_list[] = array('name' => 'Not determined', 'count' => $sensor_undet);

// Identified users by OS bucket (Apple Silicon / Intel Mac / Windows), from the
// telemetry rather than the GitHub feed. Shown as a percentage split.
$plat_map = array();
foreach ($user_plat as $uid => $lab) { if ($lab !== '') $plat_map[$lab][$uid] = true; }
$platform_list = ranked($plat_map);

// Windows users who pulled the GPU (CUDA) package: a Windows install with >=1 GPU
// run. Mac's GPU is built in (not a download), so this is Windows-only.
$windows_gpu = 0; $windows_users = 0;
foreach ($user_plat as $uid => $lab) {
    if ($lab === 'Windows') {
        $windows_users++;
        if (isset($gpu_users[$uid])) $windows_gpu++;
    }
}

// Versions in use: one vote per user, their latest reported build.
$ver = array();
foreach ($user_version as $uid => $v) $ver[$v][$uid] = true;
$current_version = '';
foreach (array_keys($ver) as $vk) {
    if ($current_version === '' || version_compare($vk, $current_version, '>')) $current_version = $vk;
}
// Ordered by version number, newest first (not by user count): the list reads
// as a release timeline, so adoption of the newest build is visible at the top.
$ver_list = array();
foreach ($ver as $vk => $set) $ver_list[] = array('name' => $vk, 'count' => count($set));
usort($ver_list, function ($a, $b) { return version_compare($b['name'], $a['name']); });

echo json_encode(array(
    'trails_cleaned'        => $BASELINE_TRAILS + $trails,
    'hours_saved'           => $BASELINE_HOURS + (int) round($trails * 30 / 3600),
    'users'                 => count($users),
    'photographers'         => count($users),
    'downloads_total'       => $BASELINE_DOWNLOADS + count($users),
    'platforms'             => $platform_list,
    'countries_count'       => count($country),
    'countries'             => $countries_list,
    'formats'               => $fmt_list,
    'orientation'           => $orient_list,
    // Measured over the SAME population as the cameras list (runs over 20 frames),
    // so the two cards describe the same set of runs.
    'no_exif_pct'           => $real_runs ? (int) round($no_exif * 100 / $real_runs) : 0,
    'cameras'               => ranked($cam),
    'camera_brands'         => $brands_list,
    'sensor'                => $sensor_list,
    'lenses'                => ranked($lens),
    'focal_lengths'         => ranked($focal, 'numeric'),
    'iso'                   => ranked($iso, 'numeric'),
    'shutter'               => ranked($shutter, 'numeric'),
    'aperture'              => ranked($aperture),
    'full_recipe'           => $recipe_list,
    'megapixels'            => $mp_list,
    'runs'                  => $real_runs,
    'avg_frames'            => $real_runs ? (int) round($real_frames / $real_runs) : 0,
    'trails_per_frame'      => $real_frames ? round($real_trails / $real_frames, 1) : 0,
    'avg_trails_per_run'    => $real_runs ? (int) round($real_trails / $real_runs) : 0,
    'avg_time_saved_sec'    => $real_runs ? (int) round($real_trails * 30 / $real_runs) : 0,
    'gpu_vs_cpu'            => $gpu_cpu_list,
    'windows_gpu'           => $windows_gpu,
    'windows_gpu_pct'       => $windows_users ? (int) round($windows_gpu * 100 / $windows_users) : 0,
    'startrails'            => $startrails,
    'timelapses'            => $timelapses,
    'versions'              => $ver_list,
    'current_version'       => $current_version,
    'generated'             => gmdate('c'),
));
