// ticktock.hpp
//

#ifndef LZZ_SIR_SBM_LZZ_ticktock_hpp
#define LZZ_SIR_SBM_LZZ_ticktock_hpp
#line 3 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//ticktock.hpp"
#include <iostream>
#include <chrono>
#define LZZ_INLINE inline
#line 6 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//ticktock.hpp"
struct TickTock
{
#line 8 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//ticktock.hpp"
  std::chrono::time_point <std::chrono::high_resolution_clock> start;
#line 8 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//ticktock.hpp"
  std::chrono::time_point <std::chrono::high_resolution_clock> end;
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//ticktock.hpp"
  std::chrono::duration <float> duration;
#line 10 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//ticktock.hpp"
  void tick ();
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//ticktock.hpp"
  void tock ();
#line 19 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//ticktock.hpp"
  void tock_print ();
};
#undef LZZ_INLINE
#endif
