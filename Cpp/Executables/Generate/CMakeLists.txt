add_executable(Bernoulli_SIR_MC Bernoulli_SIR_MC.cpp)
target_link_libraries(Bernoulli_SIR_MC PUBLIC ${DEFAULT_LIBS} OpenMP::OpenMP_CXX DataFrame Common Math)
target_include_directories(Bernoulli_SIR_MC PUBLIC ${INCLUDE_DIRS})
target_compile_options(Bernoulli_SIR_MC PUBLIC "-static")


pybind11_add_module(Bernoulli_SIR_MC_Binder Binder.cpp)
target_link_libraries(Bernoulli_SIR_MC_Binder PUBLIC ${DEFAULT_LIBS} OpenMP::OpenMP_CXX DataFrame Common Math)
target_include_directories(Bernoulli_SIR_MC_Binder PUBLIC ${INCLUDE_DIRS})

add_compile_options("-fPIC")