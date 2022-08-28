add_executable(Bernoulli_SIR_MC Bernoulli_SIR_MC.cpp)
target_link_libraries(Bernoulli_SIR_MC PRIVATE ${DEFAULT_LIBS} OpenMP::OpenMP_CXX)
target_include_directories(Bernoulli_SIR_MC PRIVATE "../../include")
target_compile_options(Bernoulli_SIR_MC PRIVATE "-static")