#ifndef SBM_SIMULATION_UTILS_VALIDATION_HPP
#define SBM_SIMULATION_UTILS_VALIDATION_HPP
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
namespace SBM_Simulation
{
void if_false_throw(bool condition, std::string msg);
}
#endif
