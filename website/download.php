<?php
// Star Trail CleanR download-button counter + redirect.
// The website's Download buttons link here with ?os=<platform>. We tick a
// per-platform count, then 302-redirect to the matching GitHub release asset.
// Called with no ?os (or ?json=1), returns the current counts as JSON so the
// totals can be checked. Counts only button presses from the site, separate
// from GitHub's own download_count (which also includes auto-updates and bots).

$DATA = '/home/dh_bmigjp/stc_data/downloads.json';
$BASE = 'https://github.com/bruceherwig-dot/star-trail-cleanr/releases/latest/download';

$ASSETS = array(
    'mac-as'    => 'StarTrailCleanR-Mac-AppleSilicon.dmg',
    'mac-intel' => 'StarTrailCleanR-Mac-Intel.dmg',
    'windows'   => 'StarTrailCleanRSetup.zip',
    'linux'     => 'StarTrailCleanR-Linux-x86_64.tar.gz',
);

$os = isset($_GET['os']) ? (string) $_GET['os'] : '';

// Read-only view: no platform (or ?json=1) -> return the running totals.
if ($os === '' || isset($_GET['json'])) {
    header('Content-Type: application/json');
    header('Access-Control-Allow-Origin: *');
    header('Cache-Control: no-store');
    $c  = @json_decode(@file_get_contents($DATA), true);
    $by = (is_array($c) && isset($c['by_os']) && is_array($c['by_os'])) ? $c['by_os'] : array();
    echo json_encode(array(
        'downloads_total' => array_sum($by),
        'by_os'           => $by,
        'updated'         => (is_array($c) && isset($c['updated'])) ? $c['updated'] : null,
    ));
    exit;
}

// Unknown platform -> send to the releases page, do not count.
if (!isset($ASSETS[$os])) {
    header('Location: https://github.com/bruceherwig-dot/star-trail-cleanr/releases/latest', true, 302);
    exit;
}

// Count the press (skip obvious bots/crawlers), atomically, then redirect.
$ua     = isset($_SERVER['HTTP_USER_AGENT']) ? strtolower($_SERVER['HTTP_USER_AGENT']) : '';
$is_bot = ($ua === '' || preg_match('/bot|crawl|spider|slurp|bingpreview|facebookexternalhit|headless|preview/', $ua));

if (!$is_bot) {
    $fp = @fopen($DATA, 'c+');
    if ($fp) {
        @flock($fp, LOCK_EX);
        $c = json_decode(stream_get_contents($fp), true);
        if (!is_array($c)) $c = array();
        if (!isset($c['by_os']) || !is_array($c['by_os'])) $c['by_os'] = array();
        $c['by_os'][$os] = (isset($c['by_os'][$os]) ? (int) $c['by_os'][$os] : 0) + 1;
        $c['updated']    = gmdate('c');
        rewind($fp);
        ftruncate($fp, 0);
        fwrite($fp, json_encode($c, JSON_UNESCAPED_SLASHES));
        @flock($fp, LOCK_UN);
        @fclose($fp);
    }
}

header('Location: ' . $BASE . '/' . $ASSETS[$os], true, 302);
exit;
