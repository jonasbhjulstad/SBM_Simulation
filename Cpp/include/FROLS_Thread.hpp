//
// Created by arch on 9/21/22.
//

#ifndef FROLS_FROLS_THREAD_HPP
#define FROLS_FROLS_THREAD_HPP
#include <execution>
namespace FROLS
{
    struct Thread_Enumerator
    {
        size_t get_num(std::thread::id id)
        {
            std::lock_guard<std::mutex> guard(m);
            auto num_ptr = threadmap.find(id);
            if (num_ptr != threadmap.end())
                return num_ptr->second;
            threadmap[id] = thread_count;
            thread_count++;
        }
    private:
        std::unordered_map<std::thread::id, size_t> threadmap;
        std::mutex m;
        size_t thread_count = 0;
    };
}

#endif //FROLS_FROLS_THREAD_HPP
