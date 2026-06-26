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

function ranked($map) {
    $out = array();
    foreach ($map as $k => $set) $out[] = array('name' => $k, 'count' => count($set));
    usort($out, function ($a, $b) { return $b['count'] - $a['count']; });
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
            if ($ctry !== '') add_user($country, $ctry, $id);
        }
        fclose($fh);
    }
}

$gh = github_downloads($GH_CACHE, $GH_TTL);

$fmt_list = array();
foreach ($fmt as $k => $n) $fmt_list[] = array('name' => $k, 'count' => $n);
usort($fmt_list, function ($a, $b) { return $b['count'] - $a['count']; });

echo json_encode(array(
    'trails_cleaned'        => $BASELINE_TRAILS + $trails,
    'hours_saved'           => $BASELINE_HOURS + (int) round($trails * 30 / 3600),
    'users'                 => count($users),
    'photographers'         => count($users),
    'downloads_total'       => $gh['total'],
    'downloads_by_platform' => $gh['by_platform'],
    'countries_count'       => count($country),
    'countries'             => ranked($country),
    'formats'               => $fmt_list,
    'no_exif_pct'           => $runs ? (int) round($no_exif * 100 / $runs) : 0,
    'cameras'               => ranked($cam),
    'lenses'                => ranked($lens),
    'focal_lengths'         => ranked($focal),
    'runs'                  => $real_runs,
    'avg_frames'            => $real_runs ? (int) round($real_frames / $real_runs) : 0,
    'trails_per_frame'      => $real_frames ? round($real_trails / $real_frames, 1) : 0,
    'gpu_pct'               => $real_runs ? (int) round($real_gpu * 100 / $real_runs) : 0,
    'timelapses'            => $timelapses,
    'generated'             => gmdate('c'),
));
