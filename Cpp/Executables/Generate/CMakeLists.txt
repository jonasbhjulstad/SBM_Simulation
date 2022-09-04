add_executable(Bernoulli_SIR_MC Bernoulli_SIR_MC.cpp)
target_link_libraries(Bernoulli_SIR_MC PUBLIC ${DEFAULT_LIBS} OpenMP::OpenMP_CXX ${FROLS_LIBRARIES})
target_include_directories(Bernoulli_SIR_MC PRIVATE ${FROLS_INCLUDE_DIRS})
target_compile_options(Bernoulli_SIR_MC PRIVATE "-static")


pybind11_add_module(Bernoulli_SIR_MC_Binder Binder.cpp)
target_link_libraries(Bernoulli_SIR_MC_Binder PUBLIC ${DEFAULT_LIBS} OpenMP::OpenMP_CXX ${FROLS_LIBRARIES})
target_include_directories(Bernoulli_SIR_MC_Binder PUBLIC ${FROLS_INCLUDE_DIRS})

