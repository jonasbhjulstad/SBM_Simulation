add_executable(Bernoulli_SIR_MC Bernoulli_SIR_MC.cpp SIR_Determinstic.cpp)
target_link_libraries(Bernoulli_SIR_MC PUBLIC ${DEFAULT_LIBS} OpenMP::OpenMP_CXX DataFrame Common FROLS_Math graphlite FROLS_Graph)

add_executable(Bernoulli_SIS_MC Bernoulli_SIS_MC.cpp)
target_link_libraries(Bernoulli_SIS_MC PUBLIC ${DEFAULT_LIBS} OpenMP::OpenMP_CXX DataFrame Common FROLS_Math graphlite FROLS_Graph)

pybind11_add_module(Bernoulli_SIR_MC_Binder Binder.cpp)
target_link_libraries(Bernoulli_SIR_MC_Binder PUBLIC ${DEFAULT_LIBS} OpenMP::OpenMP_CXX DataFrame Common FROLS_Math graphlite FROLS_Graph)

