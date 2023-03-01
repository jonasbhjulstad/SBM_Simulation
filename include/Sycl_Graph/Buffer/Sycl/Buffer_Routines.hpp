#ifndef SYCL_GRAPH_Buffer_Routines_HPP
#define SYCL_GRAPH_Buffer_Routines_HPP
#include <CL/sycl.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <tuple>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/ranges>
namespace Sycl_Graph
{
  using namespace oneapi::dpl::experimental;
  template <typename T>
  inline void buffer_copy(sycl::queue &q, sycl::buffer<T, 1> &buf, const std::vector<T> &vec)
  {
    if (vec.size() == 0)
    {
      return;
    }
    sycl::buffer<T, 1> tmp_buf(vec.data(), sycl::range<1>(vec.size()));
    q.submit([&](sycl::handler &h)
             {
    auto acc = buf.template get_access<sycl::access::mode::write>(h);
    auto tmp_acc = tmp_buf.template get_access<sycl::access::mode::read>(h);
    h.parallel_for(vec.size(), [=](sycl::id<1> i) { acc[i] = tmp_acc[i]; }); });
  }


  template <typename T, std::unsigned_integral uI_t = uint32_t>
  void buffer_add(sycl::buffer<T, 1> &dest_buf, sycl::buffer<T, 1> src_buf,
                         sycl::queue &q, uI_t offset = 0)
  {
    if (src_buf.size() == 0)
    {
      return;
    }
    if constexpr (sizeof(T) > 0)
    {
      q.submit([&](sycl::handler &h)
               {
      auto src_acc = src_buf.template get_access<sycl::access::mode::read>(h);
      auto dest_acc =
          dest_buf.template get_access<sycl::access::mode::write>(h);
      h.parallel_for(
          src_buf.size(),
          [=](sycl::id<1> i) { dest_acc[i + offset] = src_acc[i]; }); });
    }
  }

  template <std::unsigned_integral uI_t, typename ... Ts>
  void buffer_add(std::tuple<sycl::buffer<Ts, 1>...>& dest_bufs, std::tuple<sycl::buffer<Ts, 1>...>& src_bufs, sycl::queue& q, uI_t offset = 0)
  {
    (buffer_add(std::get<sycl::buffer<Ts, 1>>(dest_bufs), std::get<sycl::buffer<Ts, 1>>(src_bufs), q, offset), ...);
  }
  template <typename T, std::unsigned_integral uI_t = uint32_t>
  void buffer_add(sycl::buffer<T, 1> &dest_buf, const std::vector<T> &vec,
                       sycl::queue &q, uI_t offset = 0)
  {
    if (vec.size() == 0)
    {
      return;
    }
    if constexpr (sizeof(T) > 0)
    {
      sycl::buffer<T, 1> src_buf(vec.data(), sycl::range<1>(vec.size()));
      buffer_add(dest_buf, src_buf, q, offset);
    }
  }


  template <std::unsigned_integral uI_t, typename ... Ts>
  void buffer_add(std::tuple<sycl::buffer<Ts, 1>...>& dest_bufs, const std::tuple<std::vector<Ts>...>& src_vecs, sycl::queue& q, uI_t offset = 0)
  {
    (buffer_add(std::get<sycl::buffer<Ts, 1>>(dest_bufs), std::get<std::vector<Ts>>(src_vecs), q, offset), ...);
  }

  template <typename T, std::unsigned_integral uI_t = uint32_t>
  inline void buffer_add(std::vector<sycl::buffer<T, 1> &> &bufs,
                              const std::vector<const std::vector<T> &> &vecs,
                              sycl::queue &q, const std::vector<uI_t> &offsets)
  {
    for (uI_t i = 0; i < vecs.size(); ++i)
    {
      buffer_add(bufs[i], vecs[i], q, offsets[i]);
    }
  }

  template <typename T, std::unsigned_integral uI_t = uint32_t>
  std::vector<T> buffer_get(sycl::buffer<T, 1> &buf, sycl::queue &q,
                                  uI_t offset = 0, uI_t size = 0)
  {
    if (size == 0)
    {
      size = buf.size();
    }
    std::vector<T> res(size);
    sycl::buffer<T, 1> res_buf(res.data(), sycl::range<1>(size));
    if constexpr (sizeof(T) > 0)
    {
      auto view = ranges::all_view<T, sycl::access::mode::read>(buf);
      auto res_view = ranges::all_view<T, sycl::access::mode::write>(res_buf);
      ranges::copy(oneapi::dpl::execution::dpcpp_default, view, res_view);
    }
    return res;
  }

  template <typename T, std::unsigned_integral uI_t = uint32_t>
  std::vector<T> buffer_get(sycl::buffer<T, 1> &buf, sycl::queue &q,
                                  const std::vector<uI_t> &indices)
  {
    auto condition = [&indices](auto i) { return std::find(indices.begin(), indices.end(), i) != indices.end(); };
    return buffer_get(buf, q, condition);
  }


  template <std::unsigned_integral uI_t, typename ... Ts>
  std::tuple<std::vector<Ts>...> buffer_get(std::tuple<sycl::buffer<Ts, 1>...>& bufs, sycl::queue& q, uI_t offset = 0, uI_t size = 0)
  {
    return std::make_tuple(buffer_get(std::get<sycl::buffer<Ts, 1>>(bufs), q, offset, size)...);
  }

  template <std::unsigned_integral uI_t, typename ... Ts>
  std::tuple<std::vector<Ts>...> buffer_get(std::tuple<sycl::buffer<Ts, 1>...>& bufs, sycl::queue& q, const std::vector<uI_t>& indices)
  {
    return std::make_tuple(buffer_get(std::get<sycl::buffer<Ts, 1>>(bufs), q, indices)...);
  }

  template <std::unsigned_integral uI_t, typename ... Ts>
  std::tuple<std::vector<Ts>...> buffer_get(std::tuple<sycl::buffer<Ts, 1>...>& bufs, sycl::queue& q, auto condition)
  {
    return std::make_tuple(buffer_get(std::get<sycl::buffer<Ts, 1>>(bufs), q, condition)...);
  }

  template <typename T, std::unsigned_integral uI_t = uint32_t>
  std::vector<uI_t> buffer_get_indices(sycl::buffer<T, 1>& buf, sycl::queue& q, auto condition)
  {
    std::vector<uI_t> res(buf.size());
    sycl::buffer<uI_t, 1> res_buf(res.data(), sycl::range<1>(buf.size()));
    q.submit([&](sycl::handler& h)
    {
      auto acc = buf.template get_access<sycl::access::mode::read>(h);
      auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
      h.parallel_for(
        buf.size(), [=](sycl::id<1> i)
        {
          if (condition(i))
          {
            res_acc[i] = i;
          }
        });
    });
    std::remove_if(res.begin(), res.end(), [](auto i) { return i == 0; });
    return res;
  }

  template <std::unsigned_integral uI_t = uint32_t, typename ... Ts>
  void buffer_assign(std::tuple<sycl::buffer<Ts, 1> ...> &bufs, sycl::queue &q, const std::vector<uI_t>& indices, const std::vector<Ts>& ... vecs)
  {
    const auto buf_size = std::get<sycl::buffer<Ts, 1> ...>(bufs).size();
    auto src_bufs = std::make_tuple(std::get<sycl::buffer<Ts, 1>>(bufs).template get_access<sycl::access::mode::read>(q)...);
    q.submit([&](sycl::handler &h)
             {
      auto src_accs = std::make_tuple(std::get<sycl::buffer<Ts, 1>>(bufs).template get_access<sycl::access::mode::read>(h)...);
      auto dest_accs = std::make_tuple(std::get<sycl::buffer<Ts, 1>>(bufs).template get_access<sycl::access::mode::read_write>(h)...);
      h.parallel_for(buf_size, [=](sycl::id<1> i)
                     {
                      if(i == indices[i])
                      {
                        ((std::get<sycl::buffer<Ts, 1>>(dest_accs)[i] = std::get<sycl::buffer<Ts, 1>>(src_accs)[i]), ...);
                      }
                     });}
    );
  }

  template <typename Target_t, std::unsigned_integral uI_t = uint32_t, typename ... Ts>
  void buffer_assign(const std::vector<Target_t>& target, std::tuple<sycl::buffer<Ts, 1> ...> &bufs, sycl::queue &q, const std::vector<Ts>& ... vecs)
  {
    sycl::buffer<Target_t, 1> target_buf(target.data(), sycl::range<1>(target.size()));
    auto indices = buffer_get_indices(target_buf, q, [&](auto t) { return t == target; });
    buffer_assign(bufs, q, indices, vecs ...);
  }


  template <std::unsigned_integral uI_t = uint32_t, typename ... Ts>
  void buffer_assign_add(std::tuple<sycl::buffer<Ts, 1> ...> &bufs, sycl::queue &q, const std::vector<uI_t>& indices, const std::vector<Ts>& ... vecs, uI_t N_max = std::numeric_limits<uI_t>::max())
  {
    const auto buf_size = std::min<uI_t>(std::get<sycl::buffer<Ts, 1> ...>(bufs).size(), N_max);
    auto src_bufs = std::make_tuple(std::get<sycl::buffer<Ts, 1>>(bufs).template get_access<sycl::access::mode::read>(q)...);
    q.submit([&](sycl::handler &h)
             {
      auto src_accs = std::make_tuple(std::get<sycl::buffer<Ts, 1>>(bufs).template get_access<sycl::access::mode::read>(h)...);
      auto dest_accs = std::make_tuple(std::get<sycl::buffer<Ts, 1>>(bufs).template get_access<sycl::access::mode::read_write>(h)...);
      h.parallel_for(buf_size, [=](sycl::id<1> i)
                     {
                      if(i == indices[i])
                      {
                        ((std::get<sycl::buffer<Ts, 1>>(dest_accs)[i] = std::get<sycl::buffer<Ts, 1>>(src_accs)[i]), ...);
                      }
                     });}
    );

    std::tuple<std::vector<Ts> ...> filtered_vecs;
    ((std::get<std::vector<Ts>>(filtered_vecs).reserve(buf_size - indices.size())), ...);
    for (auto i = 0; i < buf_size - indices.size(); i++)
    {
      if (std::find(indices.begin(), indices.end(), i) == indices.end())
      {
        ((std::get<std::vector<Ts>>(filtered_vecs).push_back(std::get<std::vector<Ts>>(vecs)[i])), ...);
      }
    }
    buffer_add(bufs, filtered_vecs, q, N_max);
  }

  template <typename Target_t, std::unsigned_integral uI_t = uint32_t, typename ... Ts>
  void buffer_assign_add(const std::vector<Target_t>& target, std::tuple<sycl::buffer<Ts, 1> ...> &bufs, sycl::queue &q, const std::vector<Ts>& ... vecs, uI_t N_max = std::numeric_limits<uI_t>::max())
  {
    sycl::buffer<Target_t, 1> target_buf(target.data(), sycl::range<1>(target.size()));
    auto indices = buffer_get_indices(target_buf, q, [&](auto t) { return t == target; });
    buffer_assign(bufs, q, indices, vecs ..., N_max);
  }

  // removes elements at offset to offset+size
  template <typename T, std::unsigned_integral uI_t = uint32_t>
  void buffer_remove(sycl::buffer<T, 1> &buf, sycl::queue &q,
                            uI_t offset = 0, uI_t size = 0)
  {
    if (size == 0)
    {
      size = buf.size();
    }
    if constexpr (sizeof(T) > 0)
    {
      q.submit([&](sycl::handler &h)
               {
      auto acc = buf.template get_access<sycl::access::mode::write>(h);
      h.parallel_for(
          size, [=](sycl::id<1> i) { acc[i + offset] = acc[i + offset + size]; }); });
    }
  }

  template <std::unsigned_integral uI_t, typename ... Ts>
  void buffer_remove(std::tuple<sycl::buffer<Ts, 1>...>& bufs, sycl::queue& q, uI_t offset = 0, uI_t size = 0)
  {
    (buffer_remove(std::get<sycl::buffer<Ts, 1>>(bufs), q, offset, size), ...);
  }

  template <typename T, std::unsigned_integral uI_t = uint32_t>
  void buffer_remove(sycl::buffer<T, 1> &buf, sycl::queue &q, auto condition)
  {
    auto buf_view = ranges::all_view<T, sycl::access::mode::read_write>(buf);
    ranges::remove_if(oneapi::dpl::execution::dpcpp_default, buf_view, condition);
  }

  template <typename T, std::unsigned_integral uI_t>
  void buffer_remove(sycl::buffer<T, 1> &buf, sycl::queue &q,
                            const std::vector<uI_t> &indices)
  {
    auto condition = [indices](auto i) { return std::find(indices.begin(), indices.end(), i) != indices.end(); };
    buffer_remove(buf, q, condition);
  }

  template <typename T, std::unsigned_integral uI_t, typename ... Ts>
  void buffer_remove(std::tuple<sycl::buffer<Ts, 1>...>& bufs, sycl::queue& q, auto condition)
  {
    auto indices = buffer_get_indices(std::get<sycl::buffer<T, 1>>(bufs), q, condition);
    
  }

  template <std::unsigned_integral uI_t, typename ... Ts>
  void buffer_remove(std::tuple<sycl::buffer<Ts, 1>...>& bufs, sycl::queue& q, const std::vector<uI_t>& indices)
  {
    (buffer_remove(std::get<sycl::buffer<Ts, 1>>(bufs), q, indices), ...);
  }

  template <typename T>
  sycl::buffer<T, 1> buffer_resize(sycl::queue &q, sycl::buffer<T, 1> &buf,
                                   size_t new_size)
  {
    sycl::buffer<T, 1> new_buf(new_size);
    auto smallest_buf_size = std::min<size_t>(buf.size(), new_size);
    q.submit([&](sycl::handler &h)
             {
    auto acc = buf.template get_access<sycl::access::mode::read>(h);
    auto new_acc = new_buf.template get_access<sycl::access::mode::write>(h);
    h.parallel_for(smallest_buf_size, [=](sycl::id<1> i) { new_acc[i] = acc[i]; }); });
    return new_buf;
  }

  template <typename ... Ts>
  std::tuple<Ts ...> buffer_resize(sycl::queue& q, std::tuple<Ts ...> bufs, size_t new_size)
  {
    return std::apply([&](auto& ... buf) { return std::make_tuple(buffer_resize(q, buf, new_size) ...); }, bufs);
  }

  template <typename T, std::unsigned_integral uI_t = uint32_t>
  sycl::buffer<T, 1> buffer_combine(sycl::queue &q, sycl::buffer<T, 1> buf0,
                        sycl::buffer<T, 1> buf1, uI_t size0 = 0, uI_t size1 = 0)
  {
    if (size0 == 0)
    {
      size0 = buf0.size();
    }
    if (size1 == 0)
    {
      size1 = buf1.size();
    }
    sycl::buffer<T, 1> new_buf(size0 + size1);
    q.submit([&](sycl::handler &h)
             {
    auto acc0 = buf0.template get_access<sycl::access::mode::read>(h);
    auto new_acc = new_buf.template get_access<sycl::access::mode::write>(h);
    h.parallel_for(size0, [=](sycl::id<1> i) { new_acc[i] = acc0[i]; }); });
    q.submit([&](sycl::handler &h)
             {
    auto acc1 = buf1.template get_access<sycl::access::mode::read>(h);
    auto new_acc = new_buf.template get_access<sycl::access::mode::write>(h);
    h.parallel_for(size1, [=](sycl::id<1> i) { new_acc[i + size0] = acc1[i]; }); });
    return new_buf;
  }

  template <std::unsigned_integral uI_t = uint32_t, typename ... Ts>
  sycl::buffer<Ts ...> buffer_combine(sycl::queue& q, std::tuple<Ts ...> bufs, uI_t size0 = 0, uI_t size1 = 0)
  {
    return std::apply([&](auto& ... buf) { return buffer_combine(q, buf ..., size0, size1); }, bufs);
  }


} // namespace Sycl_Graph
#endif