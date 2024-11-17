#include <CL/sycl.hpp>

int main() {
  cl::sycl::queue q;
  std::cout << q.get_device().get_info<cl::sycl::info::device::name>()
            << std::endl;
  return 0;
}