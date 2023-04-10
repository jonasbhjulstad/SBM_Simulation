#ifndef SYCL_GRAPH_SBM_TYPES_HPP
#define SYCL_GRAPH_SBM_TYPES_HPP
#include <CL/sycl.hpp>
#include <tuple>
#include <vector>
#include <fstream>
#include <limits>
#include <cstdint>
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

  enum SIR_State : unsigned char
  {
    SIR_INDIVIDUAL_S = 0,
    SIR_INDIVIDUAL_I = 1,
    SIR_INDIVIDUAL_R = 2
  };

  template <sycl::access_mode Mode, sycl::access::target Target = sycl::access::target::device>
  struct Edge_Accessor_t;

  struct Edge_Buffer_t
  {

    Edge_Buffer_t(uint32_t N_edges, uint32_t N_communities);
    Edge_Buffer_t(uint32_t N_edges);
    sycl::buffer<uint32_t, 1> to;
    sycl::buffer<uint32_t, 1> from;
    sycl::buffer<uint32_t, 1> self;
    static constexpr uint32_t invalid_id = std::numeric_limits<uint32_t>::max();
    template <sycl::access_mode Mode>
    Edge_Accessor_t<Mode> get_access(sycl::handler &h)
    {
      return Edge_Accessor_t<Mode>(*this, h);
    }

    sycl::event fill(uint32_t val, sycl::queue &q);
  };
  template <sycl::access_mode Mode, sycl::access::target Target>
  struct Edge_Accessor_t
  {
    Edge_Accessor_t(Edge_Buffer_t &buf, sycl::handler &h);
    sycl::accessor<uint32_t, 1, Mode, Target> to;
    sycl::accessor<uint32_t, 1, Mode, Target> from;
    sycl::accessor<uint32_t, 1, Mode, Target> self;
  };
  typedef std::array<uint32_t, 3> State_t;
  struct SBM_Graph_t
  {

    SBM_Graph_t(){};
    SBM_Graph_t(const std::vector<Node_List_t> &node_lists, const std::vector<Edge_List_t> &edge_lists);

    std::vector<uint32_t> node_list;
    std::vector<Edge_t> edge_list;
    std::vector<uint32_t> community_sizes;
    std::vector<uint32_t> connection_sizes;
    std::vector<uint32_t> connection_targets;
    std::vector<uint32_t> connection_sources;
    std::vector<uint32_t> ecm;
    std::vector<uint32_t> vcm;
    uint32_t N_vertices = 0;
    uint32_t N_edges = 0;
    uint32_t N_connections = 0;
    uint32_t N_communities = 0;

    std::tuple<sycl::buffer<Edge_t>, sycl::event> create_edge_buffer(sycl::queue &q) const;
    std::tuple<sycl::buffer<uint32_t>, sycl::event> create_community_buffer(sycl::queue &q);

  private:
    void create_connection_map();
    void create_ecm();
    void create_vcm();
  };

}

#endif