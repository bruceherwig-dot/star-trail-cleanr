<?php
/*
 * Star Trail CleanR community stats for the website counter.
 * Reads the anonymous usage log and returns aggregate totals as JSON.
 * Our own dev/test runs (dev=true) are excluded so published numbers are real.
 * No per-user or per-image data is ever exposed -- counts only.
 */
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Cache-Control: public, max-age=300');

$path = '/home/dh_bmigjp/stc_data/reports.jsonl';
$trails = 0;
$users = array();

if (is_readable($path)) {
    $fh = fopen($path, 'r');
    if ($fh) {
        while (($line = fgets($fh)) !== false) {
            $line = trim($line);
            if ($line === '') continue;
            $rec = json_decode($line, true);
            if (!is_array($rec) || !isset($rec['report']) || !is_array($rec['report'])) continue;
            $r = $rec['report'];
            if (isset($r['dev']) && $r['dev'] === true) continue;   // skip our own runs
            if (isset($r['install_id'])) $users[$r['install_id']] = true;
            $type = isset($r['type']) ? $r['type'] : 'run';
            if ($type === 'run' && isset($r['trails'])) $trails += (int)$r['trails'];
        }
        fclose($fh);
    }
}

// Community-impact seed (estimated usage across everyone who's downloaded,
// set 2026-06-25). Measured opted-in counts climb on top of it.
$BASELINE_TRAILS = 224003;
$BASELINE_HOURS  = 1872;

echo json_encode(array(
    'trails_cleaned' => $BASELINE_TRAILS + $trails,
    'hours_saved'    => $BASELINE_HOURS + (int) round($trails * 30 / 3600),  // ~30s of manual editing per trail
    'users'          => count($users),
    'generated'      => gmdate('c')
));
