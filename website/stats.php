<?php
/*
 * Star Trail CleanR community stats.
 * Reads the anonymous usage log and returns aggregate totals + breakdowns as
 * JSON, for both the homepage counter and the full /stats.html page. Our own
 * dev/test runs (dev=true) are excluded so published numbers are real. No
 * per-user or per-image data is ever exposed -- counts only. Open-ended lists
 * (cameras, lenses, focal lengths, countries) are ranked by number of unique
 * photographers (install IDs) so one heavy user can't skew them.
 */
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Cache-Control: public, max-age=300');

$REPORTS  = '/home/dh_bmigjp/stc_data/reports.jsonl';
$GH_CACHE = '/home/dh_bmigjp/stc_data/gh_downloads.json';
$GH_TTL   = 21600;  // refresh GitHub download totals at most every 6 hours

// Community-impact seed (estimated usage across everyone who's downloaded,
// set 2026-06-25). Measured opted-in counts climb on top of it.
$BASELINE_TRAILS = 224003;
$BASELINE_HOURS  = 1872;

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

function fetch_github_downloads() {
    $ctx = stream_context_create(array(
        'http'  => array('timeout' => 6, 'header' => "User-Agent: StarTrailCleanR\r\nAccept: application/vnd.github+json\r\n"),
        'https' => array('timeout' => 6, 'header' => "User-Agent: StarTrailCleanR\r\nAccept: application/vnd.github+json\r\n"),
    ));
    $total = 0;
    $plat = array('Windows' => 0, 'macOS' => 0, 'Linux' => 0);
    $got = false;
    for ($page = 1; $page <= 3; $page++) {
        $url = 'https://api.github.com/repos/bruceherwig-dot/star-trail-cleanr/releases?per_page=100&page=' . $page;
        $raw = @file_get_contents($url, false, $ctx);
        if ($raw === false) break;
        $rel = json_decode($raw, true);
        if (!is_array($rel) || count($rel) === 0) break;
        $got = true;
        foreach ($rel as $r) {
            if (empty($r['assets']) || !is_array($r['assets'])) continue;
            foreach ($r['assets'] as $a) {
                $n  = strtolower($a['name']);
                $dc = (int) $a['download_count'];
                if (strpos($n, 'best.pt') !== false || strpos($n, 'model') !== false) continue;
                if (preg_match('/\.(delta|sig|sha\d*|json|xml|txt|zsync|appcast)$/', $n)) continue;
                // Match current names (.dmg / Setup.zip / .tar.gz) AND legacy .zip
                // installers (StarTrailCleanR-Mac-*.zip, -Windows.zip) from old releases.
                if (substr($n, -4) === '.dmg' || strpos($n, 'mac') !== false)          { $plat['macOS']   += $dc; $total += $dc; }
                elseif (substr($n, -7) === '.tar.gz' || strpos($n, 'linux') !== false) { $plat['Linux']   += $dc; $total += $dc; }
                elseif (strpos($n, 'setup') !== false || substr($n, -4) === '.exe' || strpos($n, 'windows') !== false || substr($n, -4) === '.zip') { $plat['Windows'] += $dc; $total += $dc; }
            }
        }
        if (count($rel) < 100) break;
    }
    if (!$got) return null;
    $by = array();
    foreach ($plat as $k => $v) if ($v > 0) $by[] = array('name' => $k, 'count' => $v);
    usort($by, function ($a, $b) { return $b['count'] - $a['count']; });
    return array('total' => $total, 'by_platform' => $by);
}

function github_downloads($cache, $ttl) {
    if (is_readable($cache) && (time() - filemtime($cache) < $ttl)) {
        $c = json_decode(@file_get_contents($cache), true);
        if (is_array($c)) return $c;
    }
    $res = fetch_github_downloads();
    if ($res !== null) { @file_put_contents($cache, json_encode($res), LOCK_EX); return $res; }
    if (is_readable($cache)) {  // fetch failed -> serve the last good cache if any
        $c = json_decode(@file_get_contents($cache), true);
        if (is_array($c)) return $c;
    }
    return array('total' => 0, 'by_platform' => array());
}

$trails = 0; $runs = 0; $timelapses = 0; $no_exif = 0;
$real_runs = 0; $real_frames = 0; $real_trails = 0; $real_gpu = 0;  // runs over 20 frames only
$users = array();
$fmt = array();
$cam = array(); $lens = array(); $focal = array(); $country = array();

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
            $ctry = isset($rec['country']) ? $rec['country'] : '';
            $type = isset($r['type']) ? $r['type'] : 'run';
            if ($type === 'timelapse') { $timelapses++; continue; }
            $runs++;
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
            $f = isset($r['input_format']) ? strtolower($r['input_format']) : '';
            if ($f !== '') { $L = format_label($f); $fmt[$L] = (isset($fmt[$L]) ? $fmt[$L] : 0) + 1; }
            if (empty($r['camera'])) $no_exif++;
            else add_user($cam, clean_camera($r['camera']), $id);
            if (!empty($r['lens'])) {
                $ln = trim($r['lens']);
                if ($ln !== '' && !preg_match('/^0+(\.0+)?\s*mm/i', $ln) && stripos($ln, 'f/0') === false) add_user($lens, $ln, $id);
            }
            if (isset($r['focal_length'])) add_user($focal, ((int) round($r['focal_length'])) . ' mm', $id);
            if ($ctry !== '') add_user($country, country_name($ctry), $id);
        }
        fclose($fh);
    }
}

$gh = github_downloads($GH_CACHE, $GH_TTL);

$fmt_list = array();
foreach ($fmt as $k => $n) $fmt_list[] = array('name' => $k, 'count' => $n);
usort($fmt_list, function ($a, $b) { return $b['count'] - $a['count']; });

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

echo json_encode(array(
    'trails_cleaned'        => $BASELINE_TRAILS + $trails,
    'hours_saved'           => $BASELINE_HOURS + (int) round($trails * 30 / 3600),
    'users'                 => count($users),
    'photographers'         => count($users),
    'downloads_total'       => $gh['total'],
    'downloads_by_platform' => $gh['by_platform'],
    'countries_count'       => count($country),
    'countries'             => $countries_list,
    'formats'               => $fmt_list,
    'no_exif_pct'           => $runs ? (int) round($no_exif * 100 / $runs) : 0,
    'cameras'               => ranked($cam),
    'lenses'                => ranked($lens),
    'focal_lengths'         => ranked($focal, 'numeric'),
    'runs'                  => $real_runs,
    'avg_frames'            => $real_runs ? (int) round($real_frames / $real_runs) : 0,
    'trails_per_frame'      => $real_frames ? round($real_trails / $real_frames, 1) : 0,
    'gpu_pct'               => $real_runs ? (int) round($real_gpu * 100 / $real_runs) : 0,
    'timelapses'            => $timelapses,
    'generated'             => gmdate('c'),
));
