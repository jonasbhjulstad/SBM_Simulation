// ticktock.hpp
//

#ifndef LZZ_ticktock_hpp
#define LZZ_ticktock_hpp
#include <iostream>
#include <chrono>
#define LZZ_INLINE inline
struct TickTock
{
  std::chrono::time_point <std::chrono::high_resolution_clock> start;
  std::chrono::time_point <std::chrono::high_resolution_clock> end;
  std::chrono::duration <float> duration;
  void tick ();
  void tock ();
  void tock_print ();
};
#undef LZZ_INLINE
#endif
