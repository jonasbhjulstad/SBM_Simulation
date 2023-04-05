#include <Sycl_Graph/SBM_types.hpp>
namespace Sycl_Graph::SBM
{

    Edge_Buffer_t::Edge_Buffer_t(uint32_t N_edges, uint32_t N_communities)
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
    Edge_Buffer_t::Edge_Buffer_t(uint32_t N_edges) : Edge_Buffer_t(N_edges, 1) {}

    template <>
    Edge_Accessor_t<sycl::access_mode::read_write>::Edge_Accessor_t(Edge_Buffer_t &buf, sycl::handler &h) : to(buf.to.template get_access<sycl::access::mode::read_write, sycl::access::target::device>(h)), from(buf.from.template get_access<sycl::access::mode::read_write, sycl::access::target::device>(h)), self(buf.self.template get_access<sycl::access::mode::read_write, sycl::access::target::device>(h)) {}

    template <>
    Edge_Accessor_t<sycl::access_mode::read>::Edge_Accessor_t(Edge_Buffer_t &buf, sycl::handler &h) : to(buf.to.template get_access<sycl::access::mode::read, sycl::access::target::device>(h)), from(buf.from.template get_access<sycl::access::mode::read, sycl::access::target::device>(h)), self(buf.self.template get_access<sycl::access::mode::read, sycl::access::target::device>(h)) {}

    template <>
    Edge_Accessor_t<sycl::access_mode::write>::Edge_Accessor_t(Edge_Buffer_t &buf, sycl::handler &h) : to(buf.to.template get_access<sycl::access::mode::write, sycl::access::target::device>(h)), from(buf.from.template get_access<sycl::access::mode::write, sycl::access::target::device>(h)), self(buf.self.template get_access<sycl::access::mode::write, sycl::access::target::device>(h)) {}

    template <>
    Edge_Accessor_t<sycl::access_mode::read_write> Edge_Buffer_t::get_access(sycl::handler &h)
    {
        return Edge_Accessor_t<sycl::access_mode::read_write>(*this, h);
    }
    template <>
    Edge_Accessor_t<sycl::access_mode::write> Edge_Buffer_t::get_access(sycl::handler &h)
    {
        return Edge_Accessor_t<sycl::access_mode::write>(*this, h);
    }
    template <>
    Edge_Accessor_t<sycl::access_mode::read> Edge_Buffer_t::get_access(sycl::handler &h)
    {
        return Edge_Accessor_t<sycl::access_mode::read>(*this, h);
    }

    sycl::event Edge_Buffer_t::fill(uint32_t val, sycl::queue &q)
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

    uint32_t SBM_Graph_t::N_vertices() const
    {
        uint32_t N = 0;
        for (auto &&nodelist : node_list)
        {
            N += nodelist.size();
        }
        return N;
    }

    uint32_t SBM_Graph_t::N_edges() const
    {
        uint32_t N = 0;
        for (auto &&edge_list : edge_lists)
        {
            N += edge_list.size();
        }
        return N;
    }

    uint32_t SBM_Graph_t::N_connections() const
    {
        return edge_lists.size();
    }

    uint32_t SBM_Graph_t::N_communities() const
    {
        return node_list.size();
    }
}
