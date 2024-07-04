module;

#include <stdexcept>
export module SIR_SBM.exceptions;

export void throw_if(bool condition, const char* msg)
{
    if(condition)
        throw std::runtime_error(msg);
}