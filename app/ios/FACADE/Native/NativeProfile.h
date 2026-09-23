#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct { uint64_t wall, cpu, thread; } FacadeProfileToken;
void facade_profile_enable(int enabled);
FacadeProfileToken facade_profile_begin(void);
void facade_profile_end(const char* name, FacadeProfileToken token);
void facade_profile_allocation(const char* name, uint64_t bytes);
const char* facade_profile_snapshot(void);
uint64_t facade_profile_now(void);
#ifdef __cplusplus
}
struct ProfileScope {
 const char* name; FacadeProfileToken token;
 explicit ProfileScope(const char* n):name(n),token(facade_profile_begin()){}
 ~ProfileScope(){facade_profile_end(name,token);}
};
#endif
