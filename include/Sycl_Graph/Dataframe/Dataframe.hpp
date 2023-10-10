#ifndef SYCL_GRAPH_DATAFRAME_DATAFRAME_HPP
#define SYCL_GRAPH_DATAFRAME_DATAFRAME_HPP
#include <Sycl_Graph/Dataframe/Dataframe_impl.hpp>
#include <Sycl_Graph/Epidemiological/SIR_Types.hpp>

extern template struct Dataframe_t<uint32_t, 1>;
extern template struct Dataframe_t<uint32_t, 2>;
extern template struct Dataframe_t<uint32_t, 3>;
extern template struct Dataframe_t<uint32_t, 4>;


extern template struct Dataframe_t<SIR_State, 1>;
extern template struct Dataframe_t<SIR_State, 2>;
extern template struct Dataframe_t<SIR_State, 3>;
extern template struct Dataframe_t<SIR_State, 4>;

extern template struct Dataframe_t<float, 1>;
extern template struct Dataframe_t<float, 2>;
extern template struct Dataframe_t<float, 3>;
extern template struct Dataframe_t<float, 4>;


#endif
