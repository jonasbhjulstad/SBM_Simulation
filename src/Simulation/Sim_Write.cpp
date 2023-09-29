
#include <Sycl_Graph/Simulation/Sim_Write.hpp>

void write_to_file(const std::vector<std::pair<uint32_t, uint32_t>>& ccm, const std::string& fname)
{
    std::ofstream f(fname);
    for(auto& [c0, c1] : ccm)
    {
        f << c0 << "," << c1 << "\n";
    }
    f.close();
}
void ccms_to_file(const std::vector<std::vector<std::pair<uint32_t, uint32_t>>>& ccms, const std::string& output_dir)
{
    auto n = 0;
    for(auto&& ccm: ccms)
    {
        write_to_file(ccm, output_dir + "/Graph_" + std::to_string(n) + "/ccm.csv");
        n++;
    }
}
