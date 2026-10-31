\\ Breadcrumb tracing header — zero-cost instrumentation for GPU ports.
\\ Copy into your project, #include, then use STRACE() before every kernel launch.
\\ Enable: set env var TRACE_BREADCRUMB=1 or pass --breadcrumb CLI flag.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
namespace kands_trace {
inline bool breadcrumb_enabled() {
    static const char* v = std::getenv("TRACE_BREADCRUMB");
    return v && (std::strcmp(v,"1") == 0);
}
inline void emit(const char* c, const char* p, const char* t, const char* e) {
    if (!breadcrumb_enabled()) return;
    auto ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    unsigned int tid = std::hash<std::thread::id>{}(std::this_thread.get_id());
    std::fprintf(stderr, "%ld\t%u\t%s\t%s\t%s\t%s\n", (long)ts, tid, c, p, t, e?e:"");
}
inline bool perf_enabled() {
    static const char* v = std::getenv("TRACE_PERF");
    return v && (std::strcmp(v,"1") == 0);
}
}
#define STRACE(c,p,t,...) do { if(::kands_trace::breadcrumb_enabled()){ \
    char _e[512]; std::snprintf(_e,sizeof(_e),__VA_ARGS__); ::kands_trace::emit(c,p,t,_e);} }while(0)
#define STRACE_SCOPE(c,p,t,...) struct _T{ ~_T(){::kands_trace::emit(c,p,"~",0);} }_t; \
    STRACE(c,p,t,__VA_ARGS__)
#define PERF_SUMMARY(c,p,o,g,h) if(::kands_trace::perf_enabled()){ \
    std::fprintf(stderr,"PERF_TRACING component=%s phase=%s ops=%lld gpu_ms=%.2f host_ms=%.2f\n",c,p,(long long)o,g,h);}
\\ Usage: STRACE("attention","dispatch","flash_attn","seq=%d layer=%d",seq,layer); before kernel launch
