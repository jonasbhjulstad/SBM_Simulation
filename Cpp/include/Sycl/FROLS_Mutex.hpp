#ifndef FROLS_MUTEX_HPP
#define FROLS_MUTEX_HPP

namespace FROLS
{
#ifdef FROLS_USE_INTEL
//     #include <oneapi/tbb/mutex.h>
//     using mutex = tbb::mutex;
//     using lock_guard = tbb::lock_guard;
    
// #else
    #include <mutex>
    using mutex = std::mutex;
    using lock_guard = std::lock_guard<std::mutex>;
#endif
}


#endif