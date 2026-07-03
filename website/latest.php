<?php
/**
 * latest.php — self-hosted update FAILSAFE for Star Trail CleanR.
 *
 * Why this exists
 * ---------------
 * Every leg of the app's update system points at a GitHub host (api.github.com
 * for the version checks, github.io for the Sparkle feed, github.com for the
 * downloads). In a country or network where GitHub is blocked, the app cannot
 * check for or fetch an update at all. This endpoint gives the app a source it
 * CAN reach — our own domain — to at least learn that an update exists and where
 * to get it.
 *
 * How it works
 * ------------
 * This server (DreamHost) is NOT in the user's blocked country, so it can reach
 * GitHub even when the user cannot. It fetches the releases list from GitHub
 * server-side, distills the two things the app needs (the latest APP release and
 * the latest MODEL release), and returns a tiny JSON blob. The result is cached
 * to a private file for CACHE_TTL seconds so we do not hammer GitHub, and if
 * GitHub is unreachable from here too, the last good cache is served instead.
 *
 * This mirrors the existing stats.php pattern (which already fetches + file-caches
 * GitHub data), so nothing new is introduced operationally.
 *
 * Response shape
 * --------------
 * {
 *   "app":   {"tag": "2.67-beta",
 *             "downloads": {"mac-as": URL, "mac-intel": URL, "windows": URL, "linux": URL},
 *             "notes": "first line of the release body"},
 *   "model": {"tag": "model-v5", "download_url": URL, "summary": "...", "credits": "..."},
 *   "generated": <unix ts of the data>,
 *   "stale": <bool: true if served from cache because GitHub was unreachable>
 * }
 * Any field that cannot be determined is omitted or null; the app treats a
 * missing field as "nothing to offer" and shows nothing.
 */

header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');

const REPO        = 'bruceherwig-dot/star-trail-cleanr';
const CACHE_FILE  = '/home/dh_bmigjp/stc_data/latest_cache.json';
const CACHE_TTL   = 3600;   // seconds a cached result is considered fresh
const GH_TIMEOUT  = 8;      // seconds to wait on GitHub before falling back to cache

// Layer 2 mirror: our own copies of the installers + model, served from our
// domain so a user who cannot reach github.com (blocked in-country) can still
// download. MIRROR_DIR is the on-disk folder; MIRROR_BASE is its public URL.
const MIRROR_DIR  = '/home/dh_bmigjp/api.startrailcleanr.com/downloads';
const MIRROR_BASE = 'https://api.startrailcleanr.com/downloads';

/** Stable, version-less installer filenames (match the app's asset constants).
 *  Used both to look for a mirrored copy and to build the mirror URL. */
function platform_files(): array {
    return [
        'mac-as'    => 'StarTrailCleanR-Mac-AppleSilicon.dmg',
        'mac-intel' => 'StarTrailCleanR-Mac-Intel.dmg',
        'windows'   => 'StarTrailCleanRSetup.zip',
        'linux'     => 'StarTrailCleanR-Linux-x86_64.tar.gz',
    ];
}

/** The mirror URL if that file is actually present on our server, else the
 *  given fallback (the GitHub URL). This is what makes the mirror safe to enable
 *  with an empty folder: a file that has not been mirrored yet simply keeps its
 *  GitHub link, and the endpoint never advertises a mirror file that is missing. */
function mirror_or(string $filename, ?string $fallback): ?string {
    if (is_readable(MIRROR_DIR . '/' . $filename)) {
        return MIRROR_BASE . '/' . $filename;
    }
    return $fallback;
}

/** Fetch a URL with the User-Agent GitHub's API requires. Returns body or null. */
function http_get(string $url): ?string {
    $ch = curl_init($url);
    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT        => GH_TIMEOUT,
        CURLOPT_FOLLOWLOCATION => true,
        CURLOPT_HTTPHEADER     => [
            'Accept: application/vnd.github+json',
            'User-Agent: StarTrailCleanR-Failsafe',
        ],
    ]);
    $body = curl_exec($ch);
    $code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);
    if ($body === false || $code < 200 || $code >= 300) {
        return null;
    }
    return $body;
}

/** First non-empty line of a release body (the app's "what's new" summary). */
function first_line(?string $body): string {
    if (!$body) return '';
    foreach (preg_split('/\r\n|\r|\n/', $body) as $ln) {
        $ln = trim($ln);
        if ($ln !== '') return $ln;
    }
    return '';
}

/** The "Credits:"-prefixed line of a release body, if any. */
function credits_line(?string $body): string {
    if (!$body) return '';
    foreach (preg_split('/\r\n|\r|\n/', $body) as $ln) {
        $ln = trim($ln);
        if (stripos($ln, 'credits:') === 0) {
            return trim(substr($ln, strpos($ln, ':') + 1));
        }
    }
    return '';
}

/** Numeric version of a "model-v<N>" tag, or null if it is not a model tag. */
function model_num(?string $tag): ?float {
    if (!$tag) return null;
    if (preg_match('/^model-v(\d+(?:\.\d+)?)/', trim($tag), $m)) {
        return (float) $m[1];
    }
    return null;
}

/** Classify an app installer asset filename into a platform key, or null. */
function platform_of(string $name): ?string {
    $n = strtolower($name);
    // Apple Silicon and Intel Macs (current .dmg names + legacy .zip names).
    if (strpos($n, 'applesilicon') !== false) return 'mac-as';
    if (strpos($n, 'intel') !== false)        return 'mac-intel';
    if (strpos($n, 'linux') !== false)        return 'linux';
    // Windows: current "StarTrailCleanRSetup.zip" and legacy "-Windows.zip".
    if (strpos($n, 'setup') !== false || strpos($n, 'windows') !== false) return 'windows';
    return null;
}

/** Build the app's failsafe payload from the GitHub releases array. */
function distill(array $releases): array {
    $out = ['app' => null, 'model' => null];

    // Latest APP release = the newest entry that is NOT a prerelease and NOT a
    // model-* tag. GitHub returns releases newest-first, so the first match wins.
    foreach ($releases as $rel) {
        if (!empty($rel['prerelease'])) continue;
        if (model_num($rel['tag_name'] ?? null) !== null) continue;
        // Collect the GitHub download URL per platform (the fallback), then
        // prefer our mirror wherever the file is actually present on our server.
        $gh = [];
        foreach (($rel['assets'] ?? []) as $a) {
            $p = platform_of($a['name'] ?? '');
            if ($p && !isset($gh[$p]) && !empty($a['browser_download_url'])) {
                $gh[$p] = $a['browser_download_url'];
            }
        }
        $downloads = [];
        foreach (platform_files() as $p => $fn) {
            $url = mirror_or($fn, $gh[$p] ?? null);
            if ($url) $downloads[$p] = $url;
        }
        $out['app'] = [
            'tag'       => $rel['tag_name'] ?? null,
            'downloads' => $downloads,
            'notes'     => first_line($rel['body'] ?? ''),
        ];
        break;
    }

    // Latest MODEL release = the highest model-v<N> tag across ALL releases
    // (model releases are published as prereleases, so do not skip those here).
    $best = null; $best_num = null;
    foreach ($releases as $rel) {
        $num = model_num($rel['tag_name'] ?? null);
        if ($num === null) continue;
        if ($best_num === null || $num > $best_num) { $best = $rel; $best_num = $num; }
    }
    if ($best !== null) {
        $pt = null;
        foreach (($best['assets'] ?? []) as $a) {
            if (str_ends_with(strtolower($a['name'] ?? ''), '.pt') && !empty($a['browser_download_url'])) {
                $pt = $a['browser_download_url'];
                break;
            }
        }
        // Prefer our mirrored best.pt when it is present on our server.
        $pt = mirror_or('best.pt', $pt);
        $out['model'] = [
            'tag'          => $best['tag_name'] ?? null,
            'download_url' => $pt,
            'summary'      => first_line($best['body'] ?? ''),
            'credits'      => credits_line($best['body'] ?? ''),
        ];
    }
    return $out;
}

// ── Serve ──────────────────────────────────────────────────────────────────
// Fresh cache -> serve it. Otherwise refetch; on GitHub failure, serve the last
// cache marked stale; if there is no cache at all, return an empty-but-valid body.
$now = time();
$cached = is_readable(CACHE_FILE) ? json_decode(file_get_contents(CACHE_FILE), true) : null;

if (is_array($cached) && isset($cached['generated']) && ($now - $cached['generated']) < CACHE_TTL) {
    echo json_encode($cached);
    exit;
}

$body = http_get('https://api.github.com/repos/' . REPO . '/releases?per_page=100');
$releases = $body !== null ? json_decode($body, true) : null;

if (!is_array($releases)) {
    // GitHub unreachable/garbage from here too. Serve the last good cache (stale)
    // if we have one, else an empty valid payload so the app just shows nothing.
    if (is_array($cached)) { $cached['stale'] = true; echo json_encode($cached); }
    else { echo json_encode(['app' => null, 'model' => null, 'generated' => $now, 'stale' => true]); }
    exit;
}

$payload = distill($releases);
$payload['generated'] = $now;
$payload['stale'] = false;

// Best-effort cache write (never fatal to the response).
@file_put_contents(CACHE_FILE, json_encode($payload), LOCK_EX);

echo json_encode($payload);
