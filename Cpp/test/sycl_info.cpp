#include <FROLS_Sycl.hpp>
#include <iostream>
#include <chrono>
template <typename T>
inline T to_milli(T timeValue)
{
    return timeValue * static_cast<T>(1e-6);
}
int main()
{
    using namespace FROLS;
    // setup sycl on gpu
    sycl::property_list propList{sycl::property::queue::enable_profiling()};
    sycl::queue q{propList};
    sycl::device dev = q.get_device();
    sycl::context ctx = q.get_context();
    sycl::platform plat = dev.get_platform();
    std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Platform: " << plat.get_info<sycl::info::platform::name>() << std::endl;
    std::cout << "Driver version: " << plat.get_info<sycl::info::platform::version>() << std::endl;
    std::cout << "Vendor: " << plat.get_info<sycl::info::platform::vendor>() << std::endl;
    std::cout << "Profile: " << plat.get_info<sycl::info::platform::profile>() << std::endl;
    std::cout << "Device version: " << dev.get_info<sycl::info::device::version>() << std::endl;

    // print memory
    std::cout << "Global memory: " << dev.get_info<sycl::info::device::global_mem_size>() << std::endl;
    std::cout << "Local memory: " << dev.get_info<sycl::info::device::local_mem_size>() << std::endl;
    std::cout << "Max work group size: " << dev.get_info<sycl::info::device::max_work_group_size>() << std::endl;
    std::cout << "Max compute units: " << dev.get_info<sycl::info::device::max_compute_units>() << std::endl;

    // set up profiling data containers
    using wall_clock_t = std::chrono::high_resolution_clock;
    using time_point_t = std::chrono::time_point<wall_clock_t>;
    size_t profiling_iters = 100;
    std::vector<sycl::event> eventList(profiling_iters);
    std::vector<time_point_t> startTimeList(profiling_iters);
    size_t N_items = 100;
    constexpr static size_t array_size = 4;
    std::array<sycl::cl_int, array_size> A = {{1, 2, 3, 4}},
                                         B = {{1, 2, 3, 4}}, C;
    std::array<sycl::cl_float, array_size> D = {{1.f, 2.f, 3.f, 4.f}},
                                           E = {{1.f, 2.f, 3.f, 4.f}}, F;
    sycl::buffer<int, 1> bufferA(A.data(), array_size);
    sycl::buffer<int, 1> bufferB(B.data(), array_size);
    sycl::buffer<float, 1> bufferD(D.data(), array_size);
    auto cgSubmissionTime = wall_clock_t::duration::zero();
    auto kernExecutionTime = wall_clock_t::duration::zero();
    
    // Submit a kernel to the queue, returns a SYCL event
    for (size_t i = 0; i < profiling_iters; ++i)
    {
        startTimeList.at(i) = wall_clock_t::now();
        eventList.at(i) = q.submit([&](sycl::handler &cgh)
                                   {
            auto accessorA = bufferA.template get_access(cgh);
            auto accessorB = bufferB.template get_access(cgh);
            auto accessorC = bufferD.template get_access(cgh);
            cgh.parallel_for<class vector_add>(sycl::range<1>(N_items), [=](sycl::id<1> index)
                                               { accessorC[index] = accessorA[index] + accessorB[index]; }); });

        const auto cgSubmissionTimePoint = eventList.at(i).template get_profiling_info<sycl::info::event_profiling::command_submit>();
        const auto startKernExecutionTimePoint =
            eventList.at(i).template get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto endKernExecutionTimePoint =
            eventList.at(i).template get_profiling_info<sycl::info::event_profiling::command_end>();

    }

}