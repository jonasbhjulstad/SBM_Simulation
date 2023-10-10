#include <Sycl_Graph/Graph/Graph.hpp>
#include <numeric>
#include <cassert>
auto make_iota(auto start, auto end)
{
    std::vector<uint32_t> result(end-start);
    std::iota(result.begin(), result.end(), start);
    return result;
}

bool is_in_connection(auto connection, auto edge)
{
    return ((connection.first == edge.first) && (connection.second == edge.second)) || ((connection.first == edge.second) && (connection.second == edge.first));
}

int main()
{
    auto v_from = make_iota(0, 10);
    auto v_to = make_iota(10, 20);
    auto edge_list = random_connect(v_from, v_to, 0.5, 0, 42);
    std::vector<uint32_t> vcm(20);
    std::fill(vcm.begin(), vcm.begin() + 10, 0);
    std::fill(vcm.begin() + 10, vcm.end(), 1);

    std::vector<std::pair<uint32_t, uint32_t>> ccm = {{0,0}, {0,1}, {1,1}};
    std::vector<uint32_t> true_ecm(edge_list.size());
    for(int i = 0; i < edge_list.size(); i++)
    {
        for(auto c_idx = 0; c_idx < ccm.size(); c_idx++)
        {
            if(is_in_connection(ccm[c_idx], edge_list[i]))
            {
                true_ecm[i] = c_idx;
            }
        }
    }
    auto ecm = ecm_from_vcm(edge_list, vcm);
    assert(std::equal(ecm.begin(), ecm.end(), true_ecm.begin(), true_ecm.end()));
    return 0;
}
