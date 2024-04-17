#pragma once
#include <sycl/sycl.hpp>
#include <SIR_SBM/buffers.hpp>
enum class SIR_State: char
{
    Susceptible = 0,
    Infected = 1,
    Recovered = 2,
    Invalid = 3
};




struct Population_Count
{
    uint32_t S, I, R;
    Population_Count(uint32_t S, uint32_t I, uint32_t R): S{S}, I{I}, R{R} {}
    Population_Count(const std::array<uint32_t, 3>& arr): S{arr[0]}, I{arr[1]}, R{arr[2]} {}
    Population_Count operator+(const Population_Count& other) const
    {
        return Population_Count{S + other.S, I + other.I, R + other.R};
    }
    uint32_t& operator[](SIR_State s)
    {
        switch(s)
        {
            case SIR_State::Susceptible:
                return S;
            case SIR_State::Infected:
                return I;
            case SIR_State::Recovered:
                return R;
            default:
                return S;
        }
    }
};

Population_Count state_to_count(SIR_State s)
{
    switch(s)
    {
        case SIR_State::Susceptible:
            return Population_Count{1, 0, 0};
        case SIR_State::Infected:
            return Population_Count{0, 1, 0};
        case SIR_State::Recovered:
            return Population_Count{0, 0, 1};
        default:
            return Population_Count{0,0,0};
    }
}

