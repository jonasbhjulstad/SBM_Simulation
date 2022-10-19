#ifndef FROLS_MUTEX_HPP
#define FROLS_MUTEX_HPP

namespace FROLS
{
    
#include <mutex>
using mutex = std::mutex;
using lock_guard = std::lock_guard<std::mutex>;
}


#endif