// exceptions.hpp
//

#ifndef LZZ_exceptions_hpp
#define LZZ_exceptions_hpp
#include <stdexcept>
#define LZZ_INLINE inline
void throw_if (bool condition, char const * msg);
#undef LZZ_INLINE
#endif
