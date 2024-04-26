// exceptions.cpp
//

#include "exceptions.hpp"
#define LZZ_INLINE inline
void throw_if (bool condition, char const * msg)
{
    if(condition)
        throw std::runtime_error(msg);
}
#undef LZZ_INLINE
