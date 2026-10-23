/*
 * breadcrumb.h — C99 zero-cost instrumentation for GPU ports.
 *
 * Agnostic tracing: works with DirectX 12, Vulkan, HIP, or CUDA backends.
 * Enable: TRACE_BREADCRUMB=1 (env var) or --breadcrumb (CLI flag)
 * Perf:   TRACE_PERF=1 (env var) or --perf-tracing (CLI flag)
 *
 * Usage in host code:
 *   #include "breadcrumb.h"
 *   STRACE("compute", "dispatch", "add_grid", "elems=%d blocks=%d", N, num_blocks);
 *   STRACE_SCOPE_BEGIN("gpu", "sync", "dispatch_compute");
 *   ... dispatch ...
 *   STRACE_SCOPE_END("gpu", "sync", "dispatch_compute");
 *
 *   PERF_SUMMARY("compute", "add_grid", num_ops, gpu_ms, host_ms);
 *
 * Copy this file into any port project. No dependencies beyond C99 + POSIX.
 */
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── Internal state ─────────────────────── */
static int _kands_trace_breadcrumb_enabled = -1;  /* -1 = unchecked */
static int _kands_trace_perf_enabled = -1;

static inline int _kands_trace_is_enabled(void)
{
    if (_kands_trace_breadcrumb_enabled == -1) {
        const char *v = getenv("TRACE_BREADCRUMB");
        _kands_trace_breadcrumb_enabled = (v && strcmp(v, "1") == 0) ? 1 : 0;
    }
    return _kands_trace_breadcrumb_enabled;
}

static inline int _kands_trace_perf_is_enabled(void)
{
    if (_kands_trace_perf_enabled == -1) {
        const char *v = getenv("TRACE_PERF");
        _kands_trace_perf_enabled = (v && strcmp(v, "1") == 0) ? 1 : 0;
    }
    return _kands_trace_perf_enabled;
}

static inline double _kands_trace_now_ns(void)
{
    struct timespec ts;
#ifdef _WIN32
    /* On Windows, use QueryPerformanceCounter for monotonic clock */
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER now;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart * 1e9;
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
#endif
}

/* ── Public macros ────────────────────────
 * STRACE: emit a single breadcrumb event
 * STRACE_SCOPE_BEGIN/END: RAII-style begin/end for C
 * PERF_SUMMARY: emit performance summary line
 */
#define STRACE(c,p,t,...) \
    do { if (_kands_trace_is_enabled()) { \
        char _kands_e[512]; \
        snprintf(_kands_e, sizeof(_kands_e), __VA_ARGS__); \
        double _t = _kands_trace_now_ns(); \
        fprintf(stderr, "%.0f\t%u\t%s\t%s\t%s\t%s\n", \
            _t, (unsigned)(uintptr_t)pthread_self(), c, p, t, _kands_e); \
    } } while (0)

/* C-compatible scope tracing (use with STRACE_SCOPE_BEGIN/END pairs) */
#define STRACE_SCOPE_BEGIN(c,p,t,...) \
    do { if (_kands_trace_is_enabled()) { \
        STRACE(c,p,t,__VA_ARGS__); \
    } } while (0)

#define STRACE_SCOPE_END(c,p,t) \
    do { if (_kands_trace_is_enabled()) { \
        STRACE(c,p,"~", "scope_end"); \
    } } while (0)

#define PERF_SUMMARY(c,p,ops,gpu_ms,host_ms) \
    do { if (_kands_trace_perf_is_enabled()) { \
        fprintf(stderr, "PERF_TRACING component=%s phase=%s ops=%lld gpu_ms=%.2f host_ms=%.2f\n", \
            c, p, (long long)(ops), (double)(gpu_ms), (double)(host_ms)); \
    } } while (0)

/* ── CLI flag parsing ─────────────────────
 * Call once at startup to check for --breadcrumb and --perf-tracing
 */
static inline void _kands_parse_trace_args(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--breadcrumb") == 0) {
            _kands_trace_breadcrumb_enabled = 1;
        }
        if (strcmp(argv[i], "--perf-tracing") == 0) {
            _kands_trace_perf_enabled = 1;
        }
    }
}
