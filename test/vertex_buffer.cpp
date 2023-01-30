#include <Sycl_Graph/Graph/Sycl/Vertex_Buffer.hpp>
#include <Sycl_Graph/Graph/Sycl/Graph.hpp>
#include <Sycl_Graph/Graph/Graph_Base.hpp>
#include <CL/sycl.hpp>
#include <Sycl_Graph/Math/math.hpp>
int main()
{
    using namespace Sycl_Graph::Sycl;
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::enable_profiling{});
    auto v = Sycl_Graph::make_vertices(Sycl_Graph::range(0, 100), Sycl_Graph::range(0,100));
    Graph<int, Void_Edge_t, int> G(q, v);

    Sycl_Graph::Sycl::Vertex_Buffer<int, int>* pv = &G.vertex_buf;
    // G.add_vertex(data, indices);
    auto a = sizeof(Sycl_Graph::Vertex_Buffer_Base<int, int, Vertex_Buffer<int, int>>);
    auto b = sizeof(Vertex_Buffer<int, int>);

    G.add_vertex(Sycl_Graph::range(0, 100), Sycl_Graph::range(0,100));

    return 0;
    
}