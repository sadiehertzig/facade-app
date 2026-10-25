// NativeProfile.h
//
// A tiny profiler shared by the C++ engine and Swift (App/NativeProfile.swift).
// Code marks a named "span" with begin/end; the profiler adds up how many times
// each span ran and how long it took. It is off by default and the app keeps it
// off (LivePhotoEngine.swift calls facade_profile_enable(0)), so a span costs
// only a function call and one atomic read. facade_profile_now is also used on
// its own as the engine's clock.
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// What facade_profile_begin hands back: wall-clock start (ns), the calling
// thread's CPU time at the start (ns), and the thread's id. All zero when
// profiling is off, which tells facade_profile_end to ignore it.
typedef struct { uint64_t wall, cpu, thread; } FacadeProfileToken;
// Turns recording on (non-zero) or off (0).
void facade_profile_enable(int enabled);
// Starts a span. Pass the token to facade_profile_end.
FacadeProfileToken facade_profile_begin(void);
// Ends a span and adds it to the totals for `name`.
void facade_profile_end(const char* name, FacadeProfileToken token);
// Records that `name` asked for `bytes` of memory (a count, not a measurement).
void facade_profile_allocation(const char* name, uint64_t bytes);
// Returns everything recorded so far as a JSON string, plus current memory
// use. The string stays valid until the next call on the same thread.
const char* facade_profile_snapshot(void);
// Monotonic clock in nanoseconds; it keeps counting while the phone sleeps.
uint64_t facade_profile_now(void);
#ifdef __cplusplus
}
// C++ helper: a span that ends automatically when the scope closes, e.g.
//     { ProfileScope p("native.forward"); ...work... }
// The constructor calls begin, the destructor calls end, so the span is
// closed even if the work throws an exception.
struct ProfileScope {
 const char* name; FacadeProfileToken token;
 explicit ProfileScope(const char* n):name(n),token(facade_profile_begin()){}
 ~ProfileScope(){facade_profile_end(name,token);}
};
#endif
