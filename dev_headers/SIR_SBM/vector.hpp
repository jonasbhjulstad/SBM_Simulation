#pragma once
#hdr
#include <SIR_SBM/common.hpp>
#include <SIR_SBM/vector.hpp>
#include <memory>
#include <cstdint>
#include <type_traits>
namespace SIR_SBM
{
    template <typename T>
    using Vec2D = std::vector<std::vector<T>>;
    template <typename T>
    using Vec3D = std::vector<std::vector<std::vector<T>>>;
}
#end

namespace SIR_SBM {

    template <typename T>
std::vector<uint32_t> get_vector_sizes(const Vec2D<T>& vecs)
{
  std::vector<uint32_t> sizes(vecs.size());
  std::transform(vecs.begin(), vecs.end(), sizes.begin(),
                 [](const std::vector<T>& vec) { return vec.size(); });
  return sizes;
}
template <typename T>
std::vector<T> vector_merge(const Vec2D<T> &vecs) {
  std::vector<T> result;
  int N = std::accumulate(
      vecs.begin(), vecs.end(), 0L,
      [](uint32_t a, const std::vector<T> &b) { return a + b.size(); });
  result.reserve(N);
  for (const auto &vec : vecs) {
    result.insert(result.end(), vec.begin(), vec.end());
  }
  return result;
}

template <typename T = uint32_t>
std::vector<T> make_iota(uint32_t N)
{
    std::vector<T> result(N);
    std::iota(result.begin(), result.end(), 0);
    return result;
}
template <typename T>
struct Vec1DView
{
    std::unique_ptr<T>& data;
    uint32_t offset;
    uint32_t N;
    uint32_t stride;
    Vec1DView(std::unique_ptr<T>& data, uint32_t offset, uint32_t N, uint32_t stride = 1) : data(data), offset(offset), N(N), stride(stride) {}

    T &operator()(uint32_t i)
    {
        return data.get()[stride*i + offset];
    }
    T operator()(uint32_t i) const
    {
        return data.get()[stride*i + offset];
    }
    Vec1DView<T> slice(uint32_t start, uint32_t end)
    {
        return Vec1DView<T>(data, end - start);
    }
    Vec1DView<T> slice(uint32_t start)
    {
        return Vec1DView<T>(data, N - start);
    }
    uint32_t size() const
    {
        return N;
    }

    Vec1DView<T> operator+=(const Vec1DView<T>& other)
    {
        for (uint32_t i = 0; i < N; i++)
        {
            data.get()[i + offset] += other(i);
        }
        return *this;
    }

    Vec1DView<T> operator+=(const std::vector<T>& other)
    {
        for (uint32_t i = 0; i < N; i++)
        {
            data.get()[i + offset] += other[i];
        }
        return *this;
    }

    operator std::vector<T>()
    {
        std::vector<T> result(N);
        for (uint32_t i = 0; i < N; i++)
        {
            result[i] = data.get()[i + offset];
        }
        return result;
    }
};

template <typename T>
struct Vec2DView
{
    std::unique_ptr<T>& data;
    uint32_t offset;
    uint32_t N0, N1;
    Vec2DView(std::unique_ptr<T>& data, uint32_t offset, uint32_t N0, uint32_t N1) : data(data), offset(offset), N0(N0), N1(N1) {}

    T &operator()(uint32_t i, uint32_t j)
    {
        return data.get()[offset + N0*i + j];
    }
    T operator()(uint32_t i, uint32_t j) const
    {
        return data.get()[offset + N0*i + j];
    }

    const Vec1DView<T> column_view(uint32_t j) const
    {
        return Vec1DView<T>(data, offset + j, N0, N1);
    }

    Vec1DView<T> operator()(uint32_t i)
    {
        return Vec1DView<T>(data, offset + i*N0);
    }

    uint32_t size() const
    {
        return N0 * N1;
    }

    operator Vec2D<T>()
    {
        Vec2D<T> result(N0);
        for (uint32_t i = 0; i < N0; i++)
        {
            result[i] = Vec1DView<T>(data, offset + i*N1, N1);
        }
        return result;
    }
};


template <typename T>
struct LinVec2D
{
    std::unique_ptr<T> data;
    uint32_t N0, N1;
    LinVec2D(uint32_t N0, uint32_t N1) : N0(N0), N1(N1)
    {
        data = std::make_unique<T>(N0 * N1);
    }
    T &operator()(uint32_t i, uint32_t j)
    {
        return data.get()[i * N1 + j];
    }
    T operator()(uint32_t i, uint32_t j) const
    {
        return data.get()[i*N1 + j];
    }
    Vec1DView<T> operator()(uint32_t row)
    {
        return Vec1DView<T>(data,  N0*row, N1);
    }
    uint32_t size() const
    {
        return N0 * N1;
    }

    operator Vec2D<T>()
    {
        Vec2D<T> result(N0, std::vector<T>(N1));
        for (uint32_t i = 0; i < N0; i++)
        {
            result[i] = Vec1DView<T>(data, i*N1, N1);
        }
        return result;
    }

};

template <typename T>
struct LinVec3D
{
    std::unique_ptr<T> data;
    uint32_t N0, N1, N2;
    LinVec3D(uint32_t N0, uint32_t N1, uint32_t N2) : N0(N0), N1(N1), N2(N2)
    {
        data = std::make_unique<T>(N0 * N1 * N2);
    }
    T &operator()(uint32_t i, uint32_t j, uint32_t k)
    {
        return data.get()[i * N1 * N2 + j * N2 + k];
    }
    T operator()(uint32_t i, uint32_t j, uint32_t k) const
    {
        return data.get()[i * N1 * N2 + j * N2 + k];
    }
    Vec2DView<T> operator()(uint32_t row)
    {
        return Vec2DView<T>(data,  N0*row, N1, N2);
    }
    uint32_t size() const
    {
        return N0 * N1 * N2;
    }

    operator Vec3D<T>()
    {
        Vec3D<T> result(N0);
        for (uint32_t i = 0; i < N0; i++)
        {
            result[i] = Vec2DView<T>(data, i*N1*N2, N1, N2);
        }
        return result;
    }

};

} // namespace SIR_SBM