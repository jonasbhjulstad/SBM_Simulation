#ifndef SYCL_GRAPH_NETWORK_SYCL_IMPL_HPP
#define SYCL_GRAPH_NETWORK_SYCL_IMPL_HPP
#include <Sycl_Graph/Math/math.hpp>
#include <Sycl_Graph/execution.hpp>
#include <random>
#include <stddef.h>
#include <vector>
#include <limits>

namespace Sycl_Graph::Sycl
{
    namespace Network_Models
    {
    using namespace Sycl_Graph::Network_Models;
    template <class Derived, typename State, typename TemporalParam>
    struct Network
    {
        std::vector<uint32_t> population_count()
        {
            return static_cast<Derived *>(this)->population_count();
        }

        void advance()
        {
            static_cast<Derived *>(this)->advance();
        }

        void read_state(){
            static_cast<Derived *>(this)->read_state();
        }

        void advance(const TemporalParam tp)
        {
            static_cast<Derived *>(this)->advance(tp);
        }

        State read_state(const TemporalParam tp){
            return static_cast<Derived *>(this)->read_state(tp);
        }

        void reset() { static_cast<Derived *>(this)->reset(); }

        //enable if Param is not void
        bool terminate(const State &x, const TemporalParam tp = TemporalParam())
        {
            return static_cast<Derived *>(this)->terminate(x, tp);
        }

        std::vector<State>
        simulate(uint32_t Nt = std::numeric_limits<uint32_t>::max(), uint32_t Nt_min = 0, const std::vector<TemporalParam> &tp = {})
        {
            std::vector<State> trajectory(Nt + 1);
            //reserve space for the trajectories
            uint32_t t = 0;
            TemporalParam tp_i = (tp.size() > 0) ? tp[0] : TemporalParam();
            trajectory[0] = read_state(tp_i);
            for (int i = 0; i < Nt; i++)
            {
                tp_i = (tp.size() > 0) ? tp[i] : TemporalParam();
                advance(tp_i);
                trajectory[i + 1] = read_state(tp[i+1]);
                if (terminate(trajectory[i + 1], tp_i))
                {
                    break;
                }
            }
            return trajectory;
        }


    };    
} // namespace Fixed
} // namespace Network_Models
#endif