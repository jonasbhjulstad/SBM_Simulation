#ifndef SYCL_GRAPH_SBM_WRITE_HPP
#define SYCL_GRAPH_SBM_WRITE_HPP
#include <Sycl_Graph/SBM_types.hpp>
#include <filesystem>
namespace Sycl_Graph::SBM
{
  template <typename T>
  void linewrite(std::ofstream &file, const std::vector<T> &iter)
  {
    for (auto &t_i_i : iter)
    {
      file << t_i_i;
      if (&t_i_i != &iter.back())
        file << ",";
      else
        file << "\n";
    }
  }

  void linewrite(std::ofstream &file, const std::vector<State_t>& state_iter);

  void linewrite(std::ofstream &file, const std::vector<Edge_t> &iter);

  template <typename T>
  std::vector<std::vector<T>> read_buffers(std::vector<sycl::buffer<T>> &bufs,
                                           std::vector<sycl::event> &events,
                                           uint32_t N, uint32_t N_elem)
  {
    auto data_vec = std::vector<std::vector<T>>(N_elem, std::vector<T>(N));
    std::transform(bufs.begin(), bufs.end(),
                   events.begin(), data_vec.begin(), [&](auto &buf, auto &event)
                   {
                   event.wait();
                   auto buf_acc =
                       buf.template get_access<sycl::access::mode::read>();
                   std::vector<T> res(N);
                   for (int i = 0; i < N; i++) {
                     res[i] = buf_acc[i];
                   }

                   return res; });
    return data_vec;
  }

  void simulate_to_file(const SBM_Graph_t &G, const SIR_SBM_Param_t &param,
                        sycl::queue &q, const std::string &file_path,
                        uint32_t sim_idx, uint32_t seed = 42);
  void parallel_simulate_to_file(const SBM_Graph_t &G,
                                 const std::vector<SIR_SBM_Param_t> &params,
                                 std::vector<sycl::queue> &qs,
                                 const std::string &file_path, uint32_t N_sim,
                                 uint32_t seed = 42);

  void parallel_simulate_to_file(
      const std::vector<SBM_Graph_t> &Gs,
      const std::vector<std::vector<SIR_SBM_Param_t>> &params,
      std::vector<std::vector<sycl::queue>> &qs,
      const std::vector<std::string> &file_paths, uint32_t seed = 42);

}
#endif // SYCL_GRAPH_SBM_WRITE_HPP