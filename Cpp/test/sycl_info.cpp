#include <FROLS_sycl.hpp>


int main()
{
    using namespace FROLS;
    //setup sycl on gpu
    sycl::queue q;
    sycl::device dev = q.get_device();
    sycl::context ctx = q.get_context();
    sycl::platform plat = dev.get_platform();
    std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Platform: " << plat.get_info<sycl::info::platform::name>() << std::endl;
    std::cout << "Driver version: " << plat.get_info<sycl::info::platform::version>() << std::endl;
    std::cout << "Vendor: " << plat.get_info<sycl::info::platform::vendor>() << std::endl;
    std::cout << "Profile: " << plat.get_info<sycl::info::platform::profile>() << std::endl;
    std::cout << "Device version: " << dev.get_info<sycl::info::device::version>() << std::endl;

    //print memory
    std::cout << "Global memory: " << dev.get_info<sycl::info::device::global_mem_size>() << std::endl;
    std::cout << "Local memory: " << dev.get_info<sycl::info::device::local_mem_size>() << std::endl;
    std::cout << "Max work group size: " << dev.get_info<sycl::info::device::max_work_group_size>() << std::endl;
    std::cout << "Max compute units: " << dev.get_info<sycl::info::device::max_compute_units>() << std::endl;


}