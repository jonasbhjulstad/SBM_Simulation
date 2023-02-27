#include <Sycl_Graph/Graph/Graph_Base.hpp>
#include <Sycl_Graph/Graph/Sycl/Vertex_Buffer.hpp>
#include <Sycl_Graph/Graph/Sycl/Edge_Buffer.hpp>

using namespace Sycl_Graph;
typedef Vertex_Buffer<double> Double_VB;
typedef Edge_Buffer<double> Double_EB;

typedef Vertex_Buffer<int> Int_VB;
typedef Edge_Buffer<int> Int_EB;

typedef Invariant_Vertex_Buffer<Double_VB, Int_VB> IVB;
typedef Invariant_Edge_Buffer<Double_EB, Int_EB> IEB;

typedef Invariant_Graph_Base<IVB, IEB> Graph_t;

int main()
{
    Graph_t graph;
    return 0;
}