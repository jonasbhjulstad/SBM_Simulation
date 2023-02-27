#include <Sycl_Graph/Graph/Graph_Base.hpp>
#include <Sycl_Graph/Graph/Sycl/Vertex_Buffer.hpp>
#include <Sycl_Graph/Graph/Sycl/Edge_Buffer.hpp>
#include <Sycl_Graph/Graph/Invariant/Edge_Buffer.hpp>
#include <Sycl_Graph/Graph/Invariant/Vertex_Buffer.hpp>

using namespace Sycl_Graph;
typedef Sycl::Vertex_Buffer<double> Double_VB;
typedef Sycl::Edge_Buffer<double> Double_EB;

typedef Sycl::Vertex_Buffer<int> Int_VB;
typedef Sycl::Edge_Buffer<int> Int_EB;

typedef Sycl_Graph::Invariant::Vertex_Buffer<Double_VB, Int_VB> IVB;
typedef Sycl_Graph::Invariant::Edge_Buffer<Double_EB, Int_EB> IEB;

typedef Sycl_Graph::Invariant::Graph_Base<IVB, IEB> Graph_t;

int main()
{
    Graph_t graph;
    return 0;
}