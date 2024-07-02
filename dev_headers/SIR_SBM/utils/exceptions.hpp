#pragma once
#hdr
#include <stdexcept>
#end
void throw_if(bool condition, const char* msg)
{
    if(condition)
        throw std::runtime_error(msg);
}