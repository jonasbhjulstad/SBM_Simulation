#include <SIR_SBM/sycl/queue_select.hpp>
#include <SIR_SBM/utils/random.hpp>
#include <SIR_SBM/sycl/sycl_routines.hpp>
#include <oneapi/dpl/random>

using namespace SIR_SBM;
int main() {
  // sycl queue
  sycl::queue q{SIR_SBM::default_queue()};
  // get work group size
  auto work_group_size =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();
  uint32_t seed = 123;
  sycl::buffer<oneapi::dpl::ranlux48> rngs(work_group_size * 10);
  buffer_copy(q, rngs,
              generate_rngs<oneapi::dpl::ranlux48>(seed, work_group_size * 10));

  std::vector<float> nums_vec(work_group_size * 10, 0.0f);
  {
    sycl::buffer<float> nums(nums_vec.data(), work_group_size * 10);
    q.submit([&](sycl::handler &h) {
      auto num_acc = nums.get_access<sycl::access::mode::write>(h);
      auto rng_acc = rngs.get_access<sycl::access::mode::read_write>(h);
      h.parallel_for(work_group_size, [=](sycl::id<1> idx) {
        oneapi::dpl::ranlux48 rng = rng_acc[idx];
        for (int i = 0; i < 10; i++) {
          num_acc[idx * 10 + i] = rng();
        }
      });
    });
  }
  return 0;
}