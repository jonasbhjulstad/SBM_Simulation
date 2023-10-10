#include <Sycl_Graph/Dataframe/Dataframe_impl.hpp>
#include <Sycl_Graph/Epidemiological/SIR_Types.hpp>
template struct Dataframe_t<uint32_t, 1>;
template struct Dataframe_t<uint32_t, 2>;
template struct Dataframe_t<uint32_t, 3>;
template struct Dataframe_t<uint32_t, 4>;


template struct Dataframe_t<SIR_State, 1>;
template struct Dataframe_t<SIR_State, 2>;
template struct Dataframe_t<SIR_State, 3>;
template struct Dataframe_t<SIR_State, 4>;

template struct Dataframe_t<float, 1>;
template struct Dataframe_t<float, 2>;
template struct Dataframe_t<float, 3>;
template struct Dataframe_t<float, 4>;
