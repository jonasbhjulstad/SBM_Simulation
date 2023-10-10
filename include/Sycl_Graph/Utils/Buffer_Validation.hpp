#ifndef BUFFER_VALIDATION_HPP
#define BUFFER_VALIDATION_HPP
template <typename T, std::size_t N, sycl::access::mode Mode>
sycl::accessor<T,N, Mode> construct_validate_accessor(sycl::buffer<T, N>& buf, sycl::handler& h, sycl::range<N> range, sycl::range<N> offset)
{
    for(int i = 0; i < N; i++)
    {
        assert(buf.get_range()[i] >= range[i] + offset[i]);
        assert(range[i] > 0);
    }
    return sycl::accessor<T, N, Mode, sycl::access::target::global_buffer>(buf, h, range, offset);
}
template <typename T, std::size_t N, sycl::access::mode Mode>
sycl::accessor<T,N, Mode> construct_validate_accessor(sycl::buffer<T, N>& buf, sycl::handler& h, sycl::range<N> range)
{
    constexpr std::array<std::size_t, N> zeros = {};

    return std::apply([&](auto ... zs){return sycl::accessor<T, N, Mode>(buf, h, range, sycl::range<N>(zs ...));}, zeros);
}


#endif
