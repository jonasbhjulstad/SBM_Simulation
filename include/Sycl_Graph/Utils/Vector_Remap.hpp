#ifndef VECTOR_REMAP_HPP
#define VECTOR_REMAP_HPP
#include <Sycl_Graph/Utils/Vector_Remap_impl.hpp>

extern template Graphseries_t<uint32_t> remap_linear_data(uint32_t Ng, uint32_t N_sims, uint32_t Nt, uint32_t N_columns, const std::vector<uint32_t>& linear_data);
extern template Graphseries_t<SIR_State> remap_linear_data(uint32_t Ng, uint32_t N_sims, uint32_t Nt, uint32_t N_columns, const std::vector<SIR_State>& linear_data);
extern template Graphseries_t<State_t> remap_linear_data(uint32_t Ng, uint32_t N_sims, uint32_t Nt, uint32_t N_columns, const std::vector<State_t>& linear_data);


extern template Graphseries_t<uint32_t> get_N_timesteps(const Graphseries_t<uint32_t>& ts, uint32_t Nt);
extern template Graphseries_t<State_t> get_N_timesteps(const Graphseries_t<State_t>& ts, uint32_t Nt);

#endif
