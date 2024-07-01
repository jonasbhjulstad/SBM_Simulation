// exceptions.cpp
//

#include "exceptions.hpp"
#define LZZ_INLINE inline
#line 5 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//exceptions.hpp"
void throw_if (bool condition, char const * msg)
#line 6 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//exceptions.hpp"
{
    if(condition)
        throw std::runtime_error(msg);
}
#undef LZZ_INLINE
