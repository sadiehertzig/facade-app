// NativeProfile.cpp
//
// Implementation of the tiny profiler declared in NativeProfile.h.
//
// How it works. A global map holds one Row of totals per span name. When
// profiling is off (always, in the shipping app) begin() returns an all-zero
// token and end() returns at once, so a span costs two function calls and one
// relaxed atomic load.
// When it is on, end() takes a mutex and adds the span's times to its Row.
//
// Two kinds of time are kept. Wall time is real elapsed time. CPU time is how
// long the calling thread actually ran. CPU time can only be measured if
// begin and end happen on the same thread; Swift async code can resume on a
// different thread, and those spans are counted in cpuMissing instead.
//
// Span times are inclusive: an outer span's time includes its inner spans, so
// totals for different names must not be added together.
#include "NativeProfile.h"
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <sys/resource.h>
#include <time.h>
#include <atomic>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <algorithm>
// File-private state (an unnamed namespace is the C++ way to say "static").
namespace {
 // Master switch, read without locking on every begin().
 std::atomic<bool> enabled{false};
 // Totals for one span name: number of calls, total wall ns, total CPU ns,
 // calls whose CPU time could not be measured, the longest single call, and for
 // allocation records the total and the largest requested byte counts.
 struct Row {uint64_t count=0,wall=0,cpu=0,cpuMissing=0,maxWall=0,bytes=0,maxBytes=0;};
 // All rows by name, and the mutex that guards them.
 std::map<std::string,Row> rows; std::mutex lock;
 // CPU time used so far by the calling thread, in ns (0 if the clock fails).
 uint64_t cpuNow(){timespec t{};return clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t)==0 ? uint64_t(t.tv_sec)*1000000000+t.tv_nsec:0;}
 // A unique id for the calling thread.
 uint64_t threadNow(){uint64_t id=0;pthread_threadid_np(nullptr,&id);return id;}
}
// Nanoseconds from mach_continuous_time, which counts in hardware "ticks"
// and keeps counting during sleep. The timebase (numer/denom) converts ticks
// to ns; it is fetched once. The multiply is done in 128 bits so a large tick
// count cannot overflow before the divide.
uint64_t facade_profile_now(){
 static const mach_timebase_info_data_t scale=[](){mach_timebase_info_data_t s;mach_timebase_info(&s);return s;}();
 auto ticks=mach_continuous_time();return uint64_t((__uint128_t(ticks)*scale.numer)/scale.denom);
}
// Switches recording on or off for every thread.
void facade_profile_enable(int value){enabled.store(value!=0);}
// Starts a span: an all-zero token when off, otherwise the current wall time,
// thread CPU time and thread id.
FacadeProfileToken facade_profile_begin(){if(!enabled.load(std::memory_order_relaxed))return {};return {facade_profile_now(),cpuNow(),threadNow()};}
// Ends a span. A zero token means profiling was off at begin(), so nothing is
// recorded.
void facade_profile_end(const char* name,FacadeProfileToken t){
 if(!t.wall)return;auto wall=facade_profile_now()-t.wall;auto cpu=cpuNow();auto thread=threadNow();
 std::lock_guard<std::mutex> guard(lock);auto& r=rows[name];r.count++;r.wall+=wall;r.maxWall=std::max(r.maxWall,wall);
 // Only trust the CPU difference if end() runs on the same thread as begin() and
 // the CPU clock worked at both ends; otherwise count the span as missing CPU.
 if(thread==t.thread && cpu>=t.cpu && t.cpu)r.cpu+=cpu-t.cpu;else r.cpuMissing++;
}
// Records a memory request. Uses the same rows map: count = number of
// requests, bytes = total bytes requested, maxBytes = largest single request.
void facade_profile_allocation(const char* name,uint64_t bytes){
 if(!enabled.load(std::memory_order_relaxed))return;
 std::lock_guard<std::mutex> guard(lock);auto& r=rows[name];r.count++;r.bytes+=bytes;r.maxBytes=std::max(r.maxBytes,bytes);
}
// Builds a JSON snapshot: resident memory, the "physical footprint" (the
// figure iOS compares with an app's memory limit), the process's total user
// and system CPU seconds, and every span's totals. A memory figure the system
// cannot provide is written as null.
const char* facade_profile_snapshot(){
 mach_task_basic_info_data_t memory{};mach_msg_type_number_t count=MACH_TASK_BASIC_INFO_COUNT;
 auto status=task_info(mach_task_self(),MACH_TASK_BASIC_INFO,(task_info_t)&memory,&count);
 task_vm_info_data_t vm{};count=TASK_VM_INFO_COUNT;auto vmStatus=task_info(mach_task_self(),TASK_VM_INFO,(task_info_t)&vm,&count);
 rusage usage{};getrusage(RUSAGE_SELF,&usage);
 // The string is built by hand. Span names are fixed strings in the code, so
 // they never contain characters that would need escaping.
 std::ostringstream s;s<<"{\"residentBytes\":";
 if(status==KERN_SUCCESS)s<<memory.resident_size;else s<<"null";
 s<<",\"physicalFootprintBytes\":";if(vmStatus==KERN_SUCCESS)s<<vm.phys_footprint;else s<<"null";
 s<<",\"processUserCPUSeconds\":"<<usage.ru_utime.tv_sec+usage.ru_utime.tv_usec/1e6<<",\"processSystemCPUSeconds\":"<<usage.ru_stime.tv_sec+usage.ru_stime.tv_usec/1e6<<",\"spans\":{";
 // From here to the end of the function the lock is held, so no span can be
 // added to `rows` while it is being written out.
 std::lock_guard<std::mutex> guard(lock);bool first=true;
 for(const auto& [name,r]:rows){if(!first)s<<",";first=false;s<<"\""<<name<<"\":{\"count\":"<<r.count<<",\"inclusiveWallNS\":"<<r.wall<<",\"callingThreadCPUNS\":"<<r.cpu<<",\"threadCPUMissingCount\":"<<r.cpuMissing<<",\"maxWallNS\":"<<r.maxWall<<",\"requestedBytes\":"<<r.bytes<<",\"maxRequestBytes\":"<<r.maxBytes<<"}";}
 // thread_local storage keeps the returned pointer valid after this function
 // returns, until the next call on the same thread.
 s<<"}}";static thread_local std::string out;out=s.str();return out.c_str();
}
