#include <Sycl_Graph/Buffer_Routines.hpp>
#include <Sycl_Graph/SBM_types.hpp>
#include <execution>
#include <itertools.hpp>
#include <numeric>
namespace Sycl_Graph::SBM
{

  Edge_Buffer_t::Edge_Buffer_t(uint32_t N_edges, uint32_t N_communities)
      : to((sycl::range<1>(N_edges))), from((sycl::range<1>(N_edges))),
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
  Edge_Buffer_t::Edge_Buffer_t(uint32_t N_edges) : Edge_Buffer_t(N_edges, 1) {}

  template <>
  Edge_Accessor_t<sycl::access_mode::read_write>::Edge_Accessor_t(
      Edge_Buffer_t &buf, sycl::handler &h)
      : to(buf.second.template get_access<sycl::access::mode::read_write,
                                      sycl::access::target::device>(h)),
        from(buf.first.template get_access<sycl::access::mode::read_write,
                                          sycl::access::target::device>(h)),
        self(buf.self.template get_access<sycl::access::mode::read_write,
                                          sycl::access::target::device>(h)) {}

  template <>
  Edge_Accessor_t<sycl::access_mode::read>::Edge_Accessor_t(Edge_Buffer_t &buf,
                                                            sycl::handler &h)
      : to(buf.second.template get_access<sycl::access::mode::read,
                                      sycl::access::target::device>(h)),
        from(buf.first.template get_access<sycl::access::mode::read,
                                          sycl::access::target::device>(h)),
        self(buf.self.template get_access<sycl::access::mode::read,
                                          sycl::access::target::device>(h)) {}

  template <>
  Edge_Accessor_t<sycl::access_mode::write>::Edge_Accessor_t(Edge_Buffer_t &buf,
                                                             sycl::handler &h)
      : to(buf.second.template get_access<sycl::access::mode::write,
                                      sycl::access::target::device>(h)),
        from(buf.first.template get_access<sycl::access::mode::write,
                                          sycl::access::target::device>(h)),
        self(buf.self.template get_access<sycl::access::mode::write,
                                          sycl::access::target::device>(h)) {}

  template <>
  Edge_Accessor_t<sycl::access_mode::write>
  Edge_Buffer_t::get_access(sycl::handler &h)
  {
    return Edge_Accessor_t<sycl::access_mode::write>(*this, h);
  }
  template <>
  Edge_Accessor_t<sycl::access_mode::read>
  Edge_Buffer_t::get_access(sycl::handler &h)
  {
    return Edge_Accessor_t<sycl::access_mode::read>(*this, h);
  }

  sycl::event Edge_Buffer_t::fill(uint32_t val, sycl::queue &q)
  {
    return q.submit([&](sycl::handler &h)
                    {
    auto to_acc = to.template get_access<sycl::access::mode::write,
                                         sycl::access::target::device>(h);
    auto from_acc = from.template get_access<sycl::access::mode::write,
                                             sycl::access::target::device>(h);
    auto self_acc = self.template get_access<sycl::access::mode::write,
                                             sycl::access::target::device>(h);
    h.parallel_for(to_acc.size(), [=](sycl::id<1> i) {
      to_acc[i] = val;
      from_acc[i] = val;
    }); });
    q.submit([&](sycl::handler &h)
             {
    auto self_acc = self.template get_access<sycl::access::mode::write,
                                             sycl::access::target::device>(h);
    h.parallel_for(self_acc.size(), [=](sycl::id<1> i) { self_acc[i] = val; }); });
  }

  SBM_Graph_t::SBM_Graph_t(const std::vector<Node_List_t> &node_lists,
                           const std::vector<Edge_List_t> &edge_lists)
  {
    N_vertices = std::accumulate(node_lists.begin(), node_lists.end(), 0,
                                 [](int sum, const Node_List_t &list)
                                 {
                                   return sum + list.size();
                                 });

    N_edges =
        std::accumulate(edge_lists.begin(), edge_lists.end(), 0,
                        [](int sum, const Edge_List_t &list)
                        {
                          return sum + list.size();
                        });
    N_communities = node_lists.size();
    N_connections = edge_lists.size();
    std::vector<uint32_t> connection_sizes(N_connections);
    std::transform(edge_lists.begin(), edge_lists.end(), connection_sizes.begin(),
                   [](const Edge_List_t &list)
                   {
                     return list.size();
                   });

    connection_community_map.resize(N_connections);
    std::vector<uint32_t> community_idx(N_communities);
    std::iota(community_idx.begin(), community_idx.end(), 0);
    uint32_t idx = 0;
    for (auto comb : iter::combinations_with_replacement(community_idx, 2))
    {
      connection_community_map[idx] = Edge_t(comb[0], comb[1], connection_sizes[idx]);
      idx++;
    }

    for(int i = 0; i < node_lists.size(); i++)
    {
      std::vector<uint32_t> idx(node_lists[i].size(), i);
      vcm.insert(vcm.end(), idx.begin(), idx.end());
    }

    for(int i = 0; i < edge_lists.size(); i++)
    {
      std::vector<uint32_t> idx(edge_lists[i].size(), i);
      ecm.insert(ecm.end(), idx.begin(), idx.end());
    }

    std::vector<Edge_t> flattened_edge_list;

    //fill with edge lists
    for (auto &list : edge_lists)
    {
      flattened_edge_list.insert(flattened_edge_list.end(), list.begin(), list.end());
    }
    this->edge_list = flattened_edge_list;


  }

  void SBM_Graph_t::remap(const std::vector<uint32_t> &map, const std::vector<Edge_t> &idx_connection_map)
  {

    std::vector<uint32_t> v_idx(N_vertices);
    std::iota(v_idx.begin(), v_idx.end(), 0);
    std::transform(v_idx.begin(), v_idx.end(), vcm.begin(),
                   [&](uint32_t &v)
                   { return map[v]; });

    std::transform(edge_list.begin(), edge_list.end(), ecm.begin(), [&](Edge_t &e)
                   {
      auto to_mapped_edge = Edge_t{map[e.first], map[e.second]};
      auto it = std::find(idx_connection_map.begin(), idx_connection_map.end(), to_mapped_edge);
      if(it != idx_connection_map.end())
        return (uint32_t)std::distance(idx_connection_map.begin(), it);
      auto from_mapped_edge = Edge_t{map[e.second], map[e.first]};
      it = std::find(idx_connection_map.begin(), idx_connection_map.end(), from_mapped_edge);
      assert(it != idx_connection_map.end() && "Could not find edge in connection map");
      return (uint32_t)std::distance(idx_connection_map.begin(), it); });

    N_connections = idx_connection_map.size();
    N_communities = *std::max_element(map.begin(), map.end()) + 1;
  }


} // namespace Sycl_Graph::SBM
