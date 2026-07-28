/*
 * The "works for other coders" experiment, in its purest form.
 *
 * Rounds 1-2 proved: feeds fine, network fine (even in the engine's own async
 * recipe), engine binary genuine, three engine versions all fail — but every
 * failing case so far hosted the engine inside a Python process (our app is
 * Python; so is the diagnostic harness). Nearly every app WinSparkle works for
 * is a normal compiled Windows program.
 *
 * This is that normal program: a tiny C host that loads the same DLL, applies
 * the same configuration in the same order as star_trail_cleanr.py, runs the
 * same no-UI check against the same feed, and reports the outcome.
 *
 *   usage:  winsparkle_c_host.exe <path-to-WinSparkle.dll> <appcast-url>
 *   exit:   0 found or up-to-date, 2 error callback fired, 3 timed out,
 *           10+ setup failures (load/getproc), each printed.
 *
 * Loads via LoadLibrary/GetProcAddress so no import library is needed and the
 * dynamic loading matches how ctypes loads it — the HOST LANGUAGE is the only
 * variable under test. Compiled on the CI runner; never shipped.
 */
#include <windows.h>
#include <stdio.h>

typedef void (*cb_t)(void);
typedef void (*set_url_t)(const wchar_t *);
typedef void (*set_details_t)(const wchar_t *, const wchar_t *, const wchar_t *);
typedef void (*set_auto_t)(int);
typedef void (*set_cb_t)(cb_t);
typedef void (*plain_t)(void);

static volatile int g_outcome = 0;   /* 1 found, 2 up-to-date, 3 error */

static void on_found(void)    { g_outcome = 1; }
static void on_notfound(void) { g_outcome = 2; }
static void on_error(void)    { g_outcome = 3; }

int wmain(int argc, wchar_t **argv)
{
    if (argc != 3) {
        fwprintf(stderr, L"usage: %ls <WinSparkle.dll> <appcast-url>\n", argv[0]);
        return 10;
    }

    HMODULE dll = LoadLibraryW(argv[1]);
    if (!dll) {
        wprintf(L"C-HOST load failed, Windows error %lu\n", GetLastError());
        return 11;
    }

    set_url_t     set_url     = (set_url_t)GetProcAddress(dll, "win_sparkle_set_appcast_url");
    set_details_t set_details = (set_details_t)GetProcAddress(dll, "win_sparkle_set_app_details");
    set_auto_t    set_auto    = (set_auto_t)GetProcAddress(dll, "win_sparkle_set_automatic_check_for_updates");
    set_cb_t      set_found   = (set_cb_t)GetProcAddress(dll, "win_sparkle_set_did_find_update_callback");
    set_cb_t      set_none    = (set_cb_t)GetProcAddress(dll, "win_sparkle_set_did_not_find_update_callback");
    set_cb_t      set_err     = (set_cb_t)GetProcAddress(dll, "win_sparkle_set_error_callback");
    plain_t       init        = (plain_t)GetProcAddress(dll, "win_sparkle_init");
    plain_t       check       = (plain_t)GetProcAddress(dll, "win_sparkle_check_update_without_ui");
    plain_t       cleanup     = (plain_t)GetProcAddress(dll, "win_sparkle_cleanup");

    if (!set_url || !set_details || !set_auto || !set_found || !set_none
            || !set_err || !init || !check) {
        wprintf(L"C-HOST missing export(s)\n");
        return 12;
    }

    /* NOTE: set_appcast_url takes a NARROW string in winsparkle.h
       (const char*), unlike the other setters. Convert. */
    {
        char url_narrow[2048];
        int i = 0;
        for (; argv[2][i] && i < 2047; i++)
            url_narrow[i] = (char)argv[2][i];   /* URLs are ASCII */
        url_narrow[i] = 0;
        ((void (*)(const char *))set_url)(url_narrow);
    }
    /* Same identity, order and values as the app and the Python harness. */
    set_details(L"Star Trail CleanR", L"Star Trail CleanR", L"2.81");
    set_auto(0);
    set_found(on_found);
    set_none(on_notfound);
    set_err(on_error);
    init();
    check();

    for (int waited = 0; waited < 450 && g_outcome == 0; waited++)
        Sleep(100);

    const wchar_t *names[] = {L"timed out", L"found", L"up-to-date", L"error"};
    wprintf(L"C-HOST %ls\n", names[g_outcome]);
    if (cleanup) cleanup();

    switch (g_outcome) {
        case 1: case 2: return 0;
        case 3: return 2;
        default: return 3;
    }
}
