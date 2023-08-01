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
      : to(buf.to.template get_access<sycl::access::mode::read_write,
                                      sycl::access::target::device>(h)),
        from(buf.from.template get_access<sycl::access::mode::read_write,
                                          sycl::access::target::device>(h)),
        self(buf.self.template get_access<sycl::access::mode::read_write,
                                          sycl::access::target::device>(h)) {}

  template <>
  Edge_Accessor_t<sycl::access_mode::read>::Edge_Accessor_t(Edge_Buffer_t &buf,
                                                            sycl::handler &h)
      : to(buf.to.template get_access<sycl::access::mode::read,
                                      sycl::access::target::device>(h)),
        from(buf.from.template get_access<sycl::access::mode::read,
                                          sycl::access::target::device>(h)),
        self(buf.self.template get_access<sycl::access::mode::read,
                                          sycl::access::target::device>(h)) {}

  template <>
  Edge_Accessor_t<sycl::access_mode::write>::Edge_Accessor_t(Edge_Buffer_t &buf,
                                                             sycl::handler &h)
      : to(buf.to.template get_access<sycl::access::mode::write,
                                      sycl::access::target::device>(h)),
        from(buf.from.template get_access<sycl::access::mode::write,
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
    community_sizes.resize(node_lists.size());
    std::transform(node_lists.begin(), node_lists.end(), community_sizes.begin(),
                   [](const Node_List_t &n)
                   { return n.size(); });
    connection_sizes.resize(edge_lists.size()*2 - node_lists.size());
    std::transform(edge_lists.begin(), edge_lists.end(), connection_sizes.begin(),
                   [](const Edge_List_t &e)
                   { return e.size(); });

    N_vertices =
        std::accumulate(community_sizes.begin(), community_sizes.end(), 0);
    N_edges =
        std::accumulate(connection_sizes.begin(), connection_sizes.end(), 0);
    N_communities = node_lists.size();
    N_connections = edge_lists.size();

    node_list.reserve(N_vertices);
    edge_list.reserve(N_edges);
    for (int i = 0; i < node_lists.size(); i++)
    {
      node_list.insert(node_list.end(), node_lists[i].begin(),
                       node_lists[i].end());
    }
    for (int i = 0; i < edge_lists.size(); i++)
    {
      edge_list.insert(edge_list.end(), edge_lists[i].begin(),
                       edge_lists[i].end());
    }
    // for each edge list:

    auto find_id_in_nodelist = [&](const uint32_t id)
    {
      auto it = std::find_if(node_lists.begin(), node_lists.end(), [&](const auto &n_list)
                             {
      auto node_it = std::find_if(n_list.begin(), n_list.end(), [&](const auto& n)
      {
        return n==id;
      });
      return node_it != n_list.end(); });
      return (uint32_t)std::distance(node_lists.begin(), it);
    };

    for (const auto &e_list : edge_lists)
    {
      auto first_edge = e_list[0];
      auto to_community_id = find_id_in_nodelist(first_edge.to);
      auto from_community_id = find_id_in_nodelist(first_edge.from);
      if (to_community_id != from_community_id)
      connection_community_map.push_back(Edge_t{from_community_id, to_community_id});
    }
    for (const auto &e_list : edge_lists)
    {
      auto first_edge = e_list[0];
      auto to_community_id = find_id_in_nodelist(first_edge.to);
      auto from_community_id = find_id_in_nodelist(first_edge.from);
      connection_community_map.push_back(Edge_t{to_community_id, from_community_id});
    }

    for(int i = 0; i < N_communities; i++)
    {
      connection_community_map.push_back(Edge_t{i, i});
    }
    std::transform(edge_lists.begin(), edge_lists.end(), connection_sizes.begin(), [&](const Edge_List_t &e_list)
                   { return e_list.size(); });

    create_ecm(connection_sizes);
    create_vcm();
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
      auto to_mapped_edge = Edge_t{map[e.from], map[e.to]};
      auto it = std::find(idx_connection_map.begin(), idx_connection_map.end(), to_mapped_edge);
      if(it != idx_connection_map.end())
        return (uint32_t)std::distance(idx_connection_map.begin(), it);
      auto from_mapped_edge = Edge_t{map[e.to], map[e.from]};
      it = std::find(idx_connection_map.begin(), idx_connection_map.end(), from_mapped_edge);
      assert(it != idx_connection_map.end() && "Could not find edge in connection map");
      return (uint32_t)std::distance(idx_connection_map.begin(), it); });

    N_connections = idx_connection_map.size();
    N_communities = *std::max_element(map.begin(), map.end()) + 1;
    connection_community_map = idx_connection_map;
  }
  void SBM_Graph_t::ccmap_write(const std::string &fpath)
  {
    std::ofstream f(fpath);
    for (const auto &e : connection_community_map)
    {
      f << e.from << " " << e.to << std::endl;
    }
  }

  void SBM_Graph_t::create_ecm(const std::vector<uint32_t> &connection_sizes)
  {
    ecm.reserve(N_edges);
    uint32_t offset = 0;
    for (int i = 0; i < N_connections; i++)
    {
      std::vector<uint32_t> con(connection_sizes[i], i);
      ecm.insert(ecm.end(), con.begin(), con.end());
    }
  }

  void SBM_Graph_t::create_vcm()
  {
    vcm.reserve(N_vertices);
    uint32_t offset = 0;
    for (int i = 0; i < N_communities; i++)
    {
      std::vector<uint32_t> com(community_sizes[i], i);
      vcm.insert(vcm.end(), com.begin(), com.end());
    }
  }
} // namespace Sycl_Graph::SBM
