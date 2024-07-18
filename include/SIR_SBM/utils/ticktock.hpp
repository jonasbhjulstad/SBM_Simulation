#pragma once

#include <chrono>
namespace SIR_SBM {
struct TickTock {
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  std::chrono::duration<float> duration;
  void tick();
  void tock();
  void tock_print();
};
} // namespace SIR_SBM