clang++ -g -fsycl -fsycl-targets=nvptx64-nvidia-cuda -I../../include -I../../build/_deps/cppitertools_repo-src/ -I../../build/_deps/tinymt_repo-src/include -I../../static -I/usr/include/eigen3 ../../test/SIR_Metapopulation_Simulation.cpp ../../static/Sycl_Graph/Math/math.cpp -o ../../build/test/SIR_Metapopulation_Simulation
#../../build/test/SIR_Metapopulation_Simulation
