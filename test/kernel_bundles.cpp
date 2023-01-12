#include <sycl/CL/sycl.hpp>
#include <iostream>
#include <chrono>
int main()
{
    // enable profiling
    sycl::property_list props{sycl::property::queue::enable_profiling()};

    sycl::queue q(props);

    using namespace std::chrono;

    // get current time
    auto t0 = high_resolution_clock::now();

    for (int i = 0; i < 1000; i++)
    {
        auto event = q.submit([&](sycl::handler &cgh)
                              { cgh.single_task<class test>([]()
                                                            {for (int i = 0; i < 100000000; i++)
        {
            int a = 0;
            a +=1;
        }; }); });
    }

    auto t1 = high_resolution_clock::now();

    std::cout << "time consumption: " << t1 - t0 << std::endl;

    auto ctx = q.get_context();

    auto t2 = high_resolution_clock::now();
    // use kernel bundle
    auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx);

    for (int i = 0; i < 1000; i++)
    {
        q.submit([&](sycl::handler &cgh)
                 {
        cgh.use_kernel_bundle(bundle);
        cgh.single_task<class bundle_test>([]() {for (int i = 0; i < 100000000; i++)
        {
            int a = 0;
            a +=1;
        };}); });
    }
    auto t3 = high_resolution_clock::now();

    std::cout << "time consumption: " << t3 - t2 << std::endl;


    

    //create a program without submitting it to queue


}
