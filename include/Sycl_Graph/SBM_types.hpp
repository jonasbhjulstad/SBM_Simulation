#ifndef SYCL_GRAPH_SBM_TYPES_HPP
#define SYCL_GRAPH_SBM_TYPES_HPP
#include <tuple>
#include <CL/sycl.hpp>
#include <vector>
namespace Sycl_Graph::SBM
{
  typedef std::tuple<sycl::buffer<uint32_t, 1>, sycl::buffer<uint32_t, 1>,
                     std::vector<uint32_t>, std::vector<uint32_t>, sycl::event>
      Iteration_Buffers_t;

  struct SIR_SBM_Param_t
  {
    std::vector<std::vector<float>> p_I;
    float p_R;
    float p_I0;
    float p_R0;
  };
  struct Edge_t
  {
    uint32_t from;
    uint32_t to;
  };

  typedef std::vector<Edge_t> Edge_List_t;

  typedef std::vector<uint32_t> Node_List_t;

  enum SIR_State: uint8_t
  {
    SIR_INDIVIDUAL_S = 0,
    SIR_INDIVIDUAL_I = 1,
    SIR_INDIVIDUAL_R = 2
  };
  template <sycl::access_mode Mode, sycl::access::target Target = sycl::access::target::device>
  struct Edge_Accessor_t
  {
    Edge_Accessor_t(sycl::handler &h) : to(h), from(h), self(h) {}
    sycl::accessor<uint32_t, 1, Mode, Target> to;
    sycl::accessor<uint32_t, 1, Mode, Target> from;
    sycl::accessor<uint32_t, 1, Mode, Target> self;
  };

  struct Edge_Buffer_t
  {

    Edge_Buffer_t(uint32_t N_edges, uint32_t N_communities)
        : to((sycl::range<1>(N_edges))),
          from((sycl::range<1>(N_edges))),
          self((sycl::range<1>(N_communities)))
    {
      auto to_acc = to.template get_access<sycl::access::mode::write>();
      auto from_acc = from.template get_access<sycl::access::mode::write>();
      auto self_acc = self.template get_access<sycl::access::mode::write>();
      for (uint32_t i = 0; i < N_edges; i++)
      {
        to_acc[i] = invalid_id;
        from_acc[i] = invalid_id;
      }
      for (uint32_t i = 0; i < N_communities; i++)
      {
        self_acc[i] = invalid_id;
      }
    }
    Edge_Buffer_t(uint32_t N_edges) : Edge_Buffer_t(N_edges, 1) {}
    sycl::buffer<uint32_t, 1> to;
    sycl::buffer<uint32_t, 1> from;
    sycl::buffer<uint32_t, 1> self;
    static constexpr uint32_t invalid_id = std::numeric_limits<uint32_t>::max();
    template <sycl::access_mode Mode>
    auto get_access(sycl::handler &h)
    {
      return Edge_Accessor_t<Mode>(h);
    }

    sycl::event fill(uint32_t val, sycl::queue &q)
    {
      return q.submit([&](sycl::handler &h)
                      {
      auto to_acc = to.template get_access<sycl::access::mode::write, sycl::access::target::device>(h);
      auto from_acc = from.template get_access<sycl::access::mode::write, sycl::access::target::device>(h);
      auto self_acc = self.template get_access<sycl::access::mode::write, sycl::access::target::device>(h);
      h.parallel_for(to_acc.size(), [=](sycl::id<1> i){
        to_acc[i] = val;
        from_acc[i] = val;
      }); });
      q.submit([&](sycl::handler &h)
               {
      auto self_acc = self.template get_access<sycl::access::mode::write, sycl::access::target::device>(h);
      h.parallel_for(self_acc.size(), [=](sycl::id<1> i){
        self_acc[i] = val;
      }); });
    }
  };

}

#endif