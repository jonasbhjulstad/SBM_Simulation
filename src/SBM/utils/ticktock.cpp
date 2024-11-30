#include <SBM/utils/ticktock.hpp>

#include <iostream>
namespace SBM {
void TickTock::tick() { start = std::chrono::high_resolution_clock::now(); }
void TickTock::tock() {
  end = std::chrono::high_resolution_clock::now();
  duration = end - start;
}
void TickTock::tock_print() {
  tock();
  std::cout << "Elapsed time[ms]: " << duration.count() * 1000.0f << std::endl;
}
} // namespace SBM