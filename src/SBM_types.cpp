#include <Sycl_Graph/Buffer_Routines.hpp>
#include <Sycl_Graph/SBM_types.hpp>
#include <itertools.hpp>
#include <numeric>
namespace Sycl_Graph::SBM {

Edge_Buffer_t::Edge_Buffer_t(uint32_t N_edges, uint32_t N_communities)
    : to((sycl::range<1>(N_edges))), from((sycl::range<1>(N_edges))),
      self((sycl::range<1>(N_communities))) {
  auto to_acc = to.template get_access<sycl::access::mode::write>();
  auto from_acc = from.template get_access<sycl::access::mode::write>();
  auto self_acc = self.template get_access<sycl::access::mode::write>();
  for (uint32_t i = 0; i < N_edges; i++) {
    to_acc[i] = invalid_id;
    from_acc[i] = invalid_id;
  }
  for (uint32_t i = 0; i < N_communities; i++) {
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
Edge_Buffer_t::get_access(sycl::handler &h) {
  return Edge_Accessor_t<sycl::access_mode::write>(*this, h);
}
template <>
Edge_Accessor_t<sycl::access_mode::read>
Edge_Buffer_t::get_access(sycl::handler &h) {
  return Edge_Accessor_t<sycl::access_mode::read>(*this, h);
}

sycl::event Edge_Buffer_t::fill(uint32_t val, sycl::queue &q) {
  return q.submit([&](sycl::handler &h) {
    auto to_acc = to.template get_access<sycl::access::mode::write,
                                         sycl::access::target::device>(h);
    auto from_acc = from.template get_access<sycl::access::mode::write,
                                             sycl::access::target::device>(h);
    auto self_acc = self.template get_access<sycl::access::mode::write,
                                             sycl::access::target::device>(h);
    h.parallel_for(to_acc.size(), [=](sycl::id<1> i) {
      to_acc[i] = val;
      from_acc[i] = val;
    });
  });
  q.submit([&](sycl::handler &h) {
    auto self_acc = self.template get_access<sycl::access::mode::write,
                                             sycl::access::target::device>(h);
    h.parallel_for(self_acc.size(), [=](sycl::id<1> i) { self_acc[i] = val; });
  });
}

SBM_Graph_t::SBM_Graph_t(const std::vector<Node_List_t> &node_lists,
                         const std::vector<Edge_List_t> &edge_lists) {
  community_sizes.resize(node_lists.size());
  std::transform(node_lists.begin(), node_lists.end(), community_sizes.begin(),
                 [](const Node_List_t &n) { return n.size(); });
  connection_sizes.resize(edge_lists.size());
  std::transform(edge_lists.begin(), edge_lists.end(), connection_sizes.begin(),
                 [](const Edge_List_t &e) { return e.size(); });

  N_vertices =
      std::accumulate(community_sizes.begin(), community_sizes.end(), 0);
  N_edges =
      std::accumulate(connection_sizes.begin(), connection_sizes.end(), 0);
  N_communities = node_lists.size();
  N_connections = edge_lists.size();

  node_list.reserve(N_vertices);
  edge_list.reserve(N_edges);
  for (int i = 0; i < node_lists.size(); i++) {
    node_list.insert(node_list.end(), node_lists[i].begin(),
                     node_lists[i].end());
  }
  for (int i = 0; i < edge_lists.size(); i++) {
    edge_list.insert(edge_list.end(), edge_lists[i].begin(),
                     edge_lists[i].end());
  }

  create_connection_map();
  create_ecm();
  create_vcm();
}

void SBM_Graph_t::remap(const std::vector<uint32_t> &map) {
  std::transform(vcm.begin(), vcm.end(), vcm.begin(),
                 [&](uint32_t &v) { return map[v]; });
  std::vector<uint32_t> connection_idx(N_connections);
  std::iota(connection_idx.begin(), connection_idx.end(), 0);

  N_communities = std::max_element(vcm.begin(), vcm.end()) - vcm.begin() + 1;
  std::vector<uint32_t> community_idx(N_communities);
  std::iota(community_idx.begin(), community_idx.end(), 0);
  uint32_t i = 0;
  std::vector<Edge_t> connection_map_new;
  std::map<Edge_t, uint32_t> connection_idx_map_new;
  for (auto &&comb : iter::combinations(community_idx, 2)) {
    connection_map_new.push_back(Edge_t{comb[0], comb[1]});
    connection_idx_map_new.insert({Edge_t{comb[0], comb[1]}, i});
    i++;}

  for (auto &&idx : community_idx) {
    connection_map_new.push_back(Edge_t{idx, idx});
    connection_idx_map_new.insert({Edge_t{idx, idx}, i});
    i++;
  }
  for (auto &&comb : iter::combinations(community_idx, 2)) {
    connection_map_new.push_back(Edge_t{comb[1], comb[0]});
    connection_idx_map_new.insert({Edge_t{comb[1], comb[0]}, i});
    i++;
  }
  std::vector<Edge_t> edge_remap(N_connections);
  std::transform(connection_map.begin(), connection_map.end(),
                 edge_remap.begin(), [&](Edge_t &e) {
                   return Edge_t{map[e.from], map[e.to]};
});
  auto remap_ecm = [&](uint32_t e) {
    auto connection_edge = connection_map[e];
    auto connection_edge_remap = edge_remap[e];
    auto connection_edge_idx = connection_idx_map_new[connection_edge];
    return connection_edge_idx;
  };
  std::transform(ecm.begin(), ecm.end(), ecm.begin(), remap_ecm);

  uint32_t N_connections_new = i;

  N_connections = N_connections_new;
  connection_map = connection_map_new;
  connection_idx_map = connection_idx_map_new;
}

void SBM_Graph_t::create_connection_map() {

  std::vector<uint32_t> community_idx(N_communities);
  std::iota(community_idx.begin(), community_idx.end(), 0);
  uint32_t i = 0;
  for (auto &&comb : iter::combinations(community_idx, 2)) {
    connection_map.push_back(Edge_t{comb[0], comb[1]});
    connection_idx_map[Edge_t{comb[0], comb[1]}] = i;
    i++;
  }
  for (auto &&idx : community_idx) {
    connection_map.push_back(Edge_t{idx, idx});
    connection_idx_map[Edge_t{idx, idx}] = i;
    i++;
  }
  for (auto &&comb : iter::combinations(community_idx, 2)) {
    connection_map.push_back(Edge_t{comb[1], comb[0]});
    connection_idx_map[Edge_t{comb[1], comb[0]}] = i;
    i++;
  }
}

void SBM_Graph_t::create_ecm() {
  ecm.reserve(N_edges);
  uint32_t offset = 0;
  for (int i = 0; i < N_connections; i++) {
    std::vector<uint32_t> con(connection_sizes[i], i);
    ecm.insert(ecm.end(), con.begin(), con.end());
  }
  
}

void SBM_Graph_t::create_vcm() {
  vcm.reserve(N_vertices);
  uint32_t offset = 0;
  for (int i = 0; i < N_communities; i++) {
    std::vector<uint32_t> com(community_sizes[i], i);
    vcm.insert(vcm.end(), com.begin(), com.end());
  }
}
} // namespace Sycl_Graph::SBM
