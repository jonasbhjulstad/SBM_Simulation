// ticktock.cpp
//

#include "ticktock.hpp"
#define LZZ_INLINE inline
#line 10 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//ticktock.hpp"
void TickTock::tick ()
#line 11 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//ticktock.hpp"
    {
        start = std::chrono::high_resolution_clock::now();
    }
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//ticktock.hpp"
void TickTock::tock ()
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//ticktock.hpp"
    {
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
    }
#line 19 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//ticktock.hpp"
void TickTock::tock_print ()
#line 20 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//ticktock.hpp"
    {
        tock();
        std::cout << "Elapsed time[ms]: " << duration.count() * 1000.0f << std::endl;
    }
#undef LZZ_INLINE
