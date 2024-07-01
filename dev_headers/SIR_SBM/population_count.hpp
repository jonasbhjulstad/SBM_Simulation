#pragma once
#hdr
#include <SIR_SBM/epidemiological.hpp>
#include <SIR_SBM/sycl_routines.hpp>
#include <SIR_SBM/vector.hpp>
#end
namespace SIR_SBM {
void validate_population(sycl::queue &q, sycl::buffer<SIR_State, 3> &state,
                         sycl::range<3> range, sycl::range<3> offset) {
#ifdef DEBUG
  int is_valid = 0;
  sycl::buffer<int> valid_buf{&is_valid, 1};
  q.submit([&](sycl::handler &h) {
    auto any_reduce = sycl::reduction(valid_buf, h, sycl::logical_or<int>());
    auto state_acc =
        state.get_access<sycl::access::mode::read>(h, range, offset);
    h.parallel_for(range, any_reduce, [=](sycl::id<3> idx, auto &reducer) {
      reducer.combine(state_acc[idx] == SIR_State::Invalid);
    });
  });
  q.wait();
  if (is_valid) {
    throw std::runtime_error("Invalid population state");
  }
#endif
}

void validate_population(sycl::queue &q, sycl::buffer<SIR_State, 3> &state) {
  validate_population(q, state, state.get_range(), sycl::range<3>(0, 0, 0));
}

sycl::event partition_population_count(sycl::queue &q,
                                       sycl::buffer<SIR_State, 3> &state,
                                       sycl::buffer<Population_Count, 3> &count,
                                       sycl::buffer<uint32_t> &vpc,
                                       uint32_t t_offset,
                                       sycl::event dep_event = {}) {

  validate_population(q, state);
  auto [N_sims, N_vertices, Nt_alloc] = get_range(state);
  Nt_alloc = std::min<uint32_t>({Nt_alloc, count.get_range()[2] - t_offset});
  return Nt_alloc <= 0 ? sycl::event{} : q.submit([&](sycl::handler &h) {
    h.depends_on(dep_event);
    auto pop_inc = [](Population_Count &pc, SIR_State s) {
      switch (s) {
      case SIR_State::Susceptible:
        pc.S++;
        break;
      case SIR_State::Infected:
        pc.I++;
        break;
      case SIR_State::Recovered:
        pc.R++;
        break;
      default:
        break;
      }
    };
    auto N_partitions = count.get_range()[0];
    auto state_acc = sycl::accessor<SIR_State, 3, sycl::access::mode::read>(
        state, h, sycl::range<3>(N_sims, N_vertices, Nt_alloc),
        sycl::id<3>(0, 0, 0));
    auto count_acc =
        sycl::accessor<Population_Count, 3, sycl::access::mode::read_write>(
            count, h, sycl::range<3>(N_sims, N_partitions, Nt_alloc),
            sycl::id<3>(0, 0, t_offset));
    auto vpc_acc = vpc.get_access<sycl::access::mode::read>(h);
    h.parallel_for(sycl::range<2>(N_sims, Nt_alloc), [=](sycl::id<2> idx) {
      uint32_t v_offset = 0;
      auto sim_idx = idx[0];
      auto t_idx = idx[1];
      for (int p_idx = 0; p_idx < N_partitions; p_idx++) {
        auto N_partition_vertices = vpc_acc[p_idx];
        for (int v_idx = v_offset; v_idx < v_offset + N_partition_vertices;
             v_idx++) {
          const SIR_State v = state_acc[sycl::id<3>(sim_idx, v_idx, t_idx)];
          pop_inc(count_acc[sycl::id<3>(sim_idx, p_idx, t_idx)], v);
        }
        v_offset += N_partition_vertices;
      }
    });
  });
}

std::vector<Population_Count>
partition_population_count(sycl::queue &q, sycl::buffer<SIR_State, 3> &state,
                           sycl::buffer<uint32_t> &vpc, uint32_t t_offset) {
  std::vector<Population_Count> count_vec(state.size());
  {
    sycl::buffer<Population_Count, 3> count{count_vec.data(),
                                            state.get_range()};
    partition_population_count(q, state, count, vpc, t_offset).wait();
  }
  return count_vec;
}

uint32_t get_new_infections(const Vec2DView<Population_Count> &pop_count,
                            uint32_t p_idx, uint32_t t_idx) {
  auto dI = pop_count(p_idx, t_idx).I - pop_count(p_idx, t_idx).I;
  auto dR = pop_count(p_idx, t_idx).R - pop_count(p_idx, t_idx).R;
  return dI + dR;
}


} // namespace SIR_SBM