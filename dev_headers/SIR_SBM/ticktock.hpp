#pragma once
#hdr
#include <iostream>
#include <chrono>
#end
struct TickTock
{
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    std::chrono::duration<float> duration;
    void tick()
    {
        start = std::chrono::high_resolution_clock::now();
    }
    void tock()
    {
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
    }
    void tock_print() 
    {
        tock();
        std::cout << "Elapsed time[ms]: " << duration.count() * 1000.0f << std::endl;
    }        
};