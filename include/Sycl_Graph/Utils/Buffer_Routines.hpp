#ifndef SYCL_GRAPH_BUFFER_ROUTINES_HPP
#define SYCL_GRAPH_BUFFER_ROUTINES_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/Common.hpp>

template <typename T>
void print_buffer(sycl::buffer<T, 1> &buf)
{
  auto acc = buf.get_host_access();
  for (int i = 0; i < buf.size(); i++)
  {
    std::cout << acc[i] << " ";
  }
  std::cout << std::endl;
}

template <typename T>
std::vector<T> read_buffer(sycl::buffer<T> &buf,
                           uint32_t N_elem = 0,
                           std::vector<sycl::event> events = {})
{
#ifdef DEBUG
  assert(buf.size() >= N_elem);
#endif
  if (N_elem == 0)
  {
    N_elem = buf.size();
  }

  std::for_each(events.begin(), events.end(), [](auto &event)
                { event.wait(); });
  auto buf_acc =
      buf.template get_access<sycl::access::mode::read>();
  std::vector<T> res(N_elem);
  for (int i = 0; i < N_elem; i++)
  {
    res[i] = buf_acc[i];
  }

  return res;
}

template <typename T>
std::vector<std::vector<T>> read_buffer(sycl::buffer<T, 2> &buf,
                           std::vector<sycl::event> events = {})
{

  std::for_each(events.begin(), events.end(), [](auto &event)
                { event.wait(); });
  auto buf_acc =
      buf.template get_access<sycl::access::mode::read>();
  auto range = buf.get_range();
  auto rows = range[0];
  auto cols = range[1];

  std::vector<std::vector<T>> res(rows, std::vector<T>(cols));
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      res[i][j] = buf_acc[i][j];
    }
  }

  return res;
}


template <typename T>
std::vector<std::vector<T>> read_buffers(std::vector<sycl::buffer<T>> &bufs,
                                         std::vector<sycl::event> &events,
                                         uint32_t N, uint32_t N_elem, sycl::queue& q)
{
#ifdef DEBUG
  assert(std::all_of(bufs.begin(), bufs.end(), [&](auto &buf)
                     { return buf.size() == N_elem; }));
  assert(bufs.size() == N);
#endif
  std::for_each(events.begin(), events.end(), [](auto &event)
                { event.wait(); });
  auto data_vec = std::vector<std::vector<T>>(N, std::vector<T>(N_elem));
  std::transform(bufs.begin(), bufs.end(),
                 events.begin(), data_vec.begin(), [&](auto &buf, auto &event)
                 {
                   std::vector<T> res(N_elem);
                   sycl::host_accessor acc(buf, sycl::read_only);
                    for (int i = 0; i < N_elem; i++)
                    {
                      res[i] = acc[i];
                    }

                   return res; });
  return data_vec;
}

template <typename T>
sycl::event copy_to_buffer(sycl::buffer<T, 1> &buf, const std::vector<T> &vec,
                           sycl::queue &q, uint32_t offset = 0, sycl::event dep_event = {})
{
#ifdef DEBUG
  assert(buf.size() >= vec.size());
#endif
  sycl::buffer<T, 1> tmp(vec.data(), sycl::range<1>(vec.size()));
  return q.submit([&](sycl::handler &h)
                  {
                    h.depends_on(dep_event);
                    auto tmp_acc = tmp.template get_access<sycl::access::mode::read>(h);
                    auto acc = buf.template get_access<sycl::access::mode::write>(h);
                    // h.parallel_for(acc.size(), [=](sycl::id<1> id)
                    //                { acc[id[0] + offset] = tmp_acc[id]; });
                    h.copy(tmp_acc, acc);
                  });
}

template <typename T>
sycl::event copy_to_buffer(sycl::buffer<T> &buf, const std::vector<std::vector<T>> &vecs, sycl::queue& q)
{
  #ifdef DEBUG
  auto tot_size = std::accumulate(vecs.begin(), vecs.end(), 0, [](auto acc, auto& vec)
                                  { return acc + vec.size(); });
  assert(buf.size() == tot_size);
  #endif
  sycl::event event;
  std::for_each(vecs.begin(), vecs.end(), [&, n = 0](const auto &vec) mutable
                 { event = copy_to_buffer(buf, vec, q, n, event);
                    n += vec.size();});
  return event;
}

template <typename T>
sycl::event buffer_fill(sycl::buffer<T, 2>& buf, const T val, sycl::queue& q)
{
  return q.submit([&](sycl::handler& h)
                  {
                    auto acc = buf.template get_access<sycl::access::mode::write>(h);
                    h.parallel_for(buf.get_range(), [=](sycl::id<2> id)
                                   { acc[id] = val; });
                  });
}


template <typename T>
auto initialize_buffer_vector(const std::vector<std::vector<T>> &init_state,
                              sycl::queue &q)
{
#ifdef DEBUG
  auto size_0 = init_state[0].size();
  assert(std::all_of(init_state.begin(), init_state.end(),
                     [size_0](const auto &vec)
                     { return vec.size() == size_0; }) &&
         "All vectors in the vector must have the same size");
#endif
  uint32_t vec_size = init_state.size();
  uint32_t buf_size = init_state[0].size();
  auto buf_vec = std::vector<sycl::buffer<T>>(
      vec_size, sycl::buffer<T>(sycl::range<1>(buf_size)));

  std::vector<sycl::event> events(vec_size);

  std::transform(buf_vec.begin(), buf_vec.end(), init_state.begin(),
                 events.begin(), [&](auto &buf, const auto &state)
                 { return copy_to_buffer(buf, state, q); });
  return std::make_tuple(buf_vec, events);
}

template <typename T>
auto initialize_buffer_vector(uint32_t buf_size, uint32_t vec_size,
                              const T init_state, sycl::queue &q)
{
  auto init_state_vec = std::vector<std::vector<T>>(
      vec_size, std::vector<T>(buf_size, init_state));
  return initialize_buffer_vector(init_state_vec, q);
}

#endif
