add_executable(Bernoulli_SIR_MC Bernoulli_SIR_MC.cpp)
target_link_libraries(Bernoulli_SIR_MC PUBLIC ${COMMON_LIBRARIES} TBB::tbb DataFrame Common FROLS_Math graphlite FROLS_Graph)

add_executable(Bernoulli_SIS_MC Bernoulli_SIS_MC.cpp)
target_link_libraries(Bernoulli_SIS_MC PUBLIC ${COMMON_LIBRARIES} TBB::tbb DataFrame Common FROLS_Math graphlite FROLS_Graph)

# pybind11_add_module(Bernoulli_SIR_MC_Binder Binder.cpp)
# target_link_libraries(Bernoulli_SIR_MC_Binder PUBLIC ${COMMON_LIBRARIES} OpenMP::OpenMP_CXX DataFrame Common FROLS_Math graphlite FROLS_Graph)

add_executable(SIR_Sine SIR_Sine.cpp)
target_link_libraries(SIR_Sine PUBLIC ${COMMON_LIBRARIES} DataFrame Common FROLS_Math ${SUNDIALS_LIBS} TBB::tbb)

add_executable(SIR_Sine_Discrete SIR_Sine_Discrete.cpp)
target_link_libraries(SIR_Sine_Discrete PUBLIC ${COMMON_LIBRARIES} DataFrame Common FROLS_Math ${SUNDIALS_LIBS} TBB::tbb)
