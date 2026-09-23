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
namespace {
 std::atomic<bool> enabled{false};
 struct Row {uint64_t count=0,wall=0,cpu=0,cpuMissing=0,maxWall=0,bytes=0,maxBytes=0;};
 std::map<std::string,Row> rows; std::mutex lock;
 uint64_t cpuNow(){timespec t{};return clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t)==0 ? uint64_t(t.tv_sec)*1000000000+t.tv_nsec:0;}
 uint64_t threadNow(){uint64_t id=0;pthread_threadid_np(nullptr,&id);return id;}
}
uint64_t facade_profile_now(){
 static const mach_timebase_info_data_t scale=[](){mach_timebase_info_data_t s;mach_timebase_info(&s);return s;}();
 auto ticks=mach_continuous_time();return uint64_t((__uint128_t(ticks)*scale.numer)/scale.denom);
}
void facade_profile_enable(int value){enabled.store(value!=0);}
FacadeProfileToken facade_profile_begin(){if(!enabled.load(std::memory_order_relaxed))return {};return {facade_profile_now(),cpuNow(),threadNow()};}
void facade_profile_end(const char* name,FacadeProfileToken t){
 if(!t.wall)return;auto wall=facade_profile_now()-t.wall;auto cpu=cpuNow();auto thread=threadNow();
 std::lock_guard<std::mutex> guard(lock);auto& r=rows[name];r.count++;r.wall+=wall;r.maxWall=std::max(r.maxWall,wall);
 if(thread==t.thread && cpu>=t.cpu && t.cpu)r.cpu+=cpu-t.cpu;else r.cpuMissing++;
}
void facade_profile_allocation(const char* name,uint64_t bytes){
 if(!enabled.load(std::memory_order_relaxed))return;
 std::lock_guard<std::mutex> guard(lock);auto& r=rows[name];r.count++;r.bytes+=bytes;r.maxBytes=std::max(r.maxBytes,bytes);
}
const char* facade_profile_snapshot(){
 mach_task_basic_info_data_t memory{};mach_msg_type_number_t count=MACH_TASK_BASIC_INFO_COUNT;
 auto status=task_info(mach_task_self(),MACH_TASK_BASIC_INFO,(task_info_t)&memory,&count);
 task_vm_info_data_t vm{};count=TASK_VM_INFO_COUNT;auto vmStatus=task_info(mach_task_self(),TASK_VM_INFO,(task_info_t)&vm,&count);
 rusage usage{};getrusage(RUSAGE_SELF,&usage);
 std::ostringstream s;s<<"{\"residentBytes\":";
 if(status==KERN_SUCCESS)s<<memory.resident_size;else s<<"null";
 s<<",\"physicalFootprintBytes\":";if(vmStatus==KERN_SUCCESS)s<<vm.phys_footprint;else s<<"null";
 s<<",\"processUserCPUSeconds\":"<<usage.ru_utime.tv_sec+usage.ru_utime.tv_usec/1e6<<",\"processSystemCPUSeconds\":"<<usage.ru_stime.tv_sec+usage.ru_stime.tv_usec/1e6<<",\"spans\":{";
 std::lock_guard<std::mutex> guard(lock);bool first=true;
 for(const auto& [name,r]:rows){if(!first)s<<",";first=false;s<<"\""<<name<<"\":{\"count\":"<<r.count<<",\"inclusiveWallNS\":"<<r.wall<<",\"callingThreadCPUNS\":"<<r.cpu<<",\"threadCPUMissingCount\":"<<r.cpuMissing<<",\"maxWallNS\":"<<r.maxWall<<",\"requestedBytes\":"<<r.bytes<<",\"maxRequestBytes\":"<<r.maxBytes<<"}";}
 s<<"}}";static thread_local std::string out;out=s.str();return out.c_str();
}
