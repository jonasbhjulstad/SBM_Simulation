// sycl_routines.hpp
//

#ifndef LZZ_SIR_SBM_LZZ_sycl_routines_hpp
#define LZZ_SIR_SBM_LZZ_sycl_routines_hpp
#line 3 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
#include <SIR_SBM/common.hpp>
#include <sycl/sycl.hpp>
#line 22 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
template <typename T, int N = 1>
void buffer_copy(sycl::queue &q, sycl::buffer<T, N> &buf,
                 const std::vector<T> &&data) {
  if (data.size() != buf.size()) {
    throw std::runtime_error("Data size is not equal to buffer size");
  }
  q.submit([&](sycl::handler &cgh) {
     auto acc = buf.template get_access<sycl::access::mode::write>(cgh);
     cgh.copy(data.data(), acc);
   }).wait();
}

template <typename T, int N = 1>
sycl::event buffer_fill(sycl::queue& q, sycl::buffer<T, N>& buf, T val) {
  return q.submit([&](sycl::handler& h) {
    auto acc = buf.template get_access<sycl::access::mode::write>(h);
    h.fill(acc, val);
  });
}

template <typename T, int N = 1>
sycl::buffer<T, N> make_buffer(sycl::queue &q, const std::vector<T> &&data,
                               sycl::range<N> r) {
  sycl::buffer<T, N> buf(r);
  q.submit([&](sycl::handler &cgh) {
     auto acc = buf.template get_access<sycl::access::mode::write>(cgh);
     cgh.copy(data.data(), acc);
   }).wait();
  return buf;
}
#define LZZ_INLINE inline
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
namespace SIR_SBM
{
#line 10 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  template <typename T, int N = 1>
#line 11 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  sycl::event buffer_copy (sycl::queue & q, sycl::buffer <T, N> & buf, std::vector <T> const & data);
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
namespace SIR_SBM
{
#line 53 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  template <int N>
#line 53 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  void validate_range (sycl::range <N> r, sycl::range <N> r_buf);
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
namespace SIR_SBM
{
#line 63 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  template <typename T>
#line 64 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  std::tuple <size_t, size_t, size_t> get_range (sycl::buffer <T, 3> const & buf);
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
namespace SIR_SBM
{
#line 70 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  template <typename T>
#line 71 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  std::tuple <size_t, size_t> get_range (sycl::buffer <T, 2> const & buf);
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
namespace SIR_SBM
{
#line 75 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  template <typename T>
#line 76 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  std::tuple <size_t> get_range (sycl::buffer <T, 1> const & buf);
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
namespace SIR_SBM
{
#line 80 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  template <typename T, size_t N>
#line 81 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  sycl::event read_buffer (sycl::queue & q, sycl::buffer <T, N> & buf, std::vector <T> & data, sycl::event dep_event);
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
namespace SIR_SBM
{
#line 92 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  sycl::event zero_fill (sycl::queue & q, sycl::buffer <uint32_t, 3> & buf, sycl::range <3> range, sycl::range <3> offset, sycl::event dep_event = {});
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
namespace SIR_SBM
{
#line 102 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  template <typename T, size_t N>
#line 103 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  std::vector <T> read_buffer (sycl::queue & q, sycl::buffer <T, N> & buf, sycl::event dep_event);
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
namespace SIR_SBM
{
#line 110 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  template <typename T>
#line 110 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  sycl::buffer <T> dummy_buf_1 ();
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
namespace SIR_SBM
{
#line 114 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  template <typename T>
#line 114 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  sycl::buffer <T, 2> dummy_buf_2 ();
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
namespace SIR_SBM
{
#line 118 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  template <typename T>
#line 118 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  sycl::buffer <T, 3> dummy_buf_3 ();
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
namespace SIR_SBM
{
#line 10 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  template <typename T, int N>
#line 11 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  sycl::event buffer_copy (sycl::queue & q, sycl::buffer <T, N> & buf, std::vector <T> const & data)
#line 12 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
                                                    {
  if (data.size() != buf.size()) {
    throw std::runtime_error("Data size is not equal to buffer size");
  }
  return q.submit([&](sycl::handler &cgh) {
    auto acc = buf.template get_access<sycl::access::mode::write>(cgh);
    cgh.copy(data.data(), acc);
  });
}
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
namespace SIR_SBM
{
#line 53 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  template <int N>
#line 53 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  void validate_range (sycl::range <N> r, sycl::range <N> r_buf)
#line 53 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
                                                                             {
  for (size_t i = 0; i < N; i++) {
    if (r[i] > r_buf[i]) {
      throw std::runtime_error("Ranges do not match at idx " +
                               std::to_string(i) + " " + std::to_string(r[i]) +
                               " " + std::to_string(r_buf[i]));
    }
  }
}
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
namespace SIR_SBM
{
#line 63 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  template <typename T>
#line 64 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  std::tuple <size_t, size_t, size_t> get_range (sycl::buffer <T, 3> const & buf)
#line 64 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
                                                                            {
  return std::make_tuple(buf.get_range()[0], buf.get_range()[1],
                         buf.get_range()[2]);
}
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
namespace SIR_SBM
{
#line 70 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  template <typename T>
#line 71 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  std::tuple <size_t, size_t> get_range (sycl::buffer <T, 2> const & buf)
#line 71 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
                                                                    {
  return std::make_tuple(buf.get_range()[0], buf.get_range()[1]);
}
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
namespace SIR_SBM
{
#line 75 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  template <typename T>
#line 76 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  std::tuple <size_t> get_range (sycl::buffer <T, 1> const & buf)
#line 76 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
                                                            {
  return std::make_tuple(buf.get_range()[0]);
}
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
namespace SIR_SBM
{
#line 80 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  template <typename T, size_t N>
#line 81 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  sycl::event read_buffer (sycl::queue & q, sycl::buffer <T, N> & buf, std::vector <T> & data, sycl::event dep_event)
#line 82 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
                                                                     {

  data.resize(buf.size());
  return q.submit([&](sycl::handler &h) {
    h.depends_on(dep_event);
    auto acc = buf.template get_access<sycl::access::mode::read>(h);
    h.copy(acc, data.data());
  });
}
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
namespace SIR_SBM
{
#line 102 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  template <typename T, size_t N>
#line 103 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  std::vector <T> read_buffer (sycl::queue & q, sycl::buffer <T, N> & buf, sycl::event dep_event)
#line 104 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
                                                  {
  std::vector<T> data;
  read_buffer<T, N>(q, buf, data, dep_event).wait();
  return data;
}
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
namespace SIR_SBM
{
#line 110 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  template <typename T>
#line 110 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  sycl::buffer <T> dummy_buf_1 ()
#line 110 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
                                                    {
  return sycl::buffer<T>(sycl::range<1>(1));
}
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
namespace SIR_SBM
{
#line 114 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  template <typename T>
#line 114 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  sycl::buffer <T, 2> dummy_buf_2 ()
#line 114 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
                                                       {
  return sycl::buffer<T, 2>(sycl::range<2>(1, 1));
}
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
namespace SIR_SBM
{
#line 118 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  template <typename T>
#line 118 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  sycl::buffer <T, 3> dummy_buf_3 ()
#line 118 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
                                                       {
  return sycl::buffer<T, 3>(sycl::range<3>(1, 1, 1));
}
}
#undef LZZ_INLINE
#endif
