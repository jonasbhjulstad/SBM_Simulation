#include <CL/sycl.hpp>
#include <iostream>



int main()
{
    sycl::queue q(sycl::gpu_selector_v);
    q.submit([&](sycl::handler& h)
    {
        sycl::stream out(1024, 256*10, h);
        h.parallel_for_work_group(sycl::range<1>(2), sycl::range<1>(32), [=](sycl::group<1> g)
        {
            out << "Hello from group " << g.get_id() << sycl::endl;
            g.parallel_for_work_item([&](sycl::h_item<1> i)
            {
                if (i.get_local_id()[0] == 2)
                out << "Hello " << i.get_global_id()[0] << ", " << i.get_local_id()[0] << sycl::endl;
                // out << "Hello from work item " << i.get_global_id() << ", " << i.get_local_id() << sycl::endl;
            });
        });
    }).wait();
    return 0;
}
