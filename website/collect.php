<?php
// Star Trail CleanR anonymous usage receiver.
// Validates a shared secret, derives country from the connection (then discards
// the IP), and appends one JSON line per cleaning-run report to a private,
// append-only log OUTSIDE the web root. Never stores images, paths, or PII.

$DATA_DIR    = '/home/dh_bmigjp/stc_data';
$LOG_FILE    = $DATA_DIR . '/reports.jsonl';
$SECRET_FILE = $DATA_DIR . '/secret.txt';
$MAX_BYTES   = 8192;

if (($_SERVER['REQUEST_METHOD'] ?? '') !== 'POST') { http_response_code(405); exit; }

$secret = @trim(@file_get_contents($SECRET_FILE));
if (!$secret) { http_response_code(500); exit; }

$sent = $_SERVER['HTTP_X_STC_KEY'] ?? '';
if (!is_string($sent) || !hash_equals($secret, $sent)) { http_response_code(403); exit; }

$raw = file_get_contents('php://input');
if ($raw === false || strlen($raw) > $MAX_BYTES) { http_response_code(413); exit; }

$data = json_decode($raw, true);
if (!is_array($data)) { http_response_code(400); exit; }

$country = '';
if (!empty($_SERVER['GEOIP_COUNTRY_CODE']))    { $country = $_SERVER['GEOIP_COUNTRY_CODE']; }
elseif (!empty($_SERVER['HTTP_CF_IPCOUNTRY'])) { $country = $_SERVER['HTTP_CF_IPCOUNTRY']; }

// Fallback: turn the connection IP into a 2-letter country code via ipwho.is
// (free, HTTPS, no key), then discard the IP. The IP is never stored, only the
// resulting country. (aiphotojudge used ipapi.co, but that is now behind a
// bot-challenge, so we use ipwho.is.) Short timeout, silent on any failure.
if ($country === '') {
    $ip = $_SERVER['REMOTE_ADDR'] ?? '';
    if ($ip !== '') {
        $ctx = stream_context_create(array(
            'http'  => array('timeout' => 3, 'header' => "User-Agent: StarTrailCleanR\r\n"),
            'https' => array('timeout' => 3, 'header' => "User-Agent: StarTrailCleanR\r\n"),
        ));
        $resp = @file_get_contents('https://ipwho.is/' . urlencode($ip) . '?fields=country_code', false, $ctx);
        if ($resp !== false) {
            $j = json_decode($resp, true);
            if (is_array($j) && isset($j['country_code']) && preg_match('/^[A-Za-z]{2}$/', $j['country_code'])) {
                $country = strtoupper($j['country_code']);
            }
        }
    }
}

$record = array('received' => gmdate('c'), 'country' => $country, 'report' => $data);
$line = json_encode($record, JSON_UNESCAPED_SLASHES) . "\n";
@file_put_contents($LOG_FILE, $line, FILE_APPEND | LOCK_EX);

http_response_code(204);
