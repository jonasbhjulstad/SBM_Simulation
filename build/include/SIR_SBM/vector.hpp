// vector.hpp
//

#ifndef LZZ_SIR_SBM_LZZ_vector_hpp
#define LZZ_SIR_SBM_LZZ_vector_hpp
#line 3 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
#include <SIR_SBM/common.hpp>
#include <memory>
#include <type_traits>
namespace SIR_SBM
{
    template <typename T>
    using Vec2D = std::vector<std::vector<T>>;
    template <typename T>
    using Vec3D = std::vector<std::vector<std::vector<T>>>;
}
#define LZZ_INLINE inline
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 16 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 17 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  std::vector <T> vector_merge (Vec2D <T> const & vecs);
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 29 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T = uint32_t>
#line 30 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  std::vector <T> make_iota (size_t N);
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 36 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 37 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  struct Vec1DView
  {
#line 39 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    std::unique_ptr <T> & data;
#line 40 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    size_t offset;
#line 41 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    size_t N;
#line 42 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    size_t stride;
#line 43 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    Vec1DView (std::unique_ptr <T> & data, size_t offset, size_t N, size_t stride = 1);
#line 45 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    T & operator () (size_t i);
#line 49 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    T operator () (size_t i) const;
#line 53 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    Vec1DView <T> slice (size_t start, size_t end);
#line 57 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    Vec1DView <T> slice (size_t start);
#line 61 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    size_t size () const;
#line 66 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    Vec1DView <T> operator += (Vec1DView <T> const & other);
#line 75 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    Vec1DView <T> operator += (std::vector <T> const & other);
#line 84 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    operator std::vector <T> ();
  };
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 95 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 96 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  struct Vec2DView
  {
#line 98 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    std::unique_ptr <T> & data;
#line 99 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    size_t offset;
#line 100 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    size_t N0;
#line 100 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    size_t N1;
#line 101 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    Vec2DView (std::unique_ptr <T> & data, size_t offset, size_t N0, size_t N1);
#line 103 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    T & operator () (size_t i, size_t j);
#line 107 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    T operator () (size_t i, size_t j) const;
#line 112 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    Vec1DView <T> column_view (size_t j);
#line 117 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    Vec1DView <T> operator () (size_t i);
#line 122 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    size_t size () const;
#line 127 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    operator Vec2D <T> ();
  };
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 139 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 140 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  struct LinVec2D
  {
#line 142 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    std::unique_ptr <T> data;
#line 143 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    size_t N0;
#line 143 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    size_t N1;
#line 144 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    LinVec3D (size_t N0, size_t N1);
#line 148 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    T & operator () (size_t i, size_t j);
#line 152 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    T operator () (size_t i, size_t j) const;
#line 156 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    Vec2DView <T> operator () (size_t row);
#line 160 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    size_t size () const;
#line 165 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    operator Vec2D <T> ();
  };
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 177 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 178 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  struct LinVec3D
  {
#line 180 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    std::unique_ptr <T> data;
#line 181 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    size_t N0;
#line 181 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    size_t N1;
#line 181 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    size_t N2;
#line 182 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    LinVec3D (size_t N0, size_t N1, size_t N2);
#line 186 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    T & operator () (size_t i, size_t j, size_t k);
#line 190 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    T operator () (size_t i, size_t j, size_t k) const;
#line 194 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    Vec2DView <T> operator () (size_t row);
#line 198 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    size_t size () const;
#line 203 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    operator Vec3D <T> ();
  };
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 16 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 17 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  std::vector <T> vector_merge (Vec2D <T> const & vecs)
#line 17 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
                                                  {
  std::vector<T> result;
  int N = std::accumulate(
      vecs.begin(), vecs.end(), 0L,
      [](size_t a, const std::vector<T> &b) { return a + b.size(); });
  result.reserve(N);
  for (const auto &vec : vecs) {
    result.insert(result.end(), vec.begin(), vec.end());
  }
  return result;
}
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 29 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 30 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  std::vector <T> make_iota (size_t N)
#line 31 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
{
    std::vector<T> result(N);
    std::iota(result.begin(), result.end(), 0);
    return result;
}
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 36 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 43 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  Vec1DView <T>::Vec1DView (std::unique_ptr <T> & data, size_t offset, size_t N, size_t stride)
#line 43 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    : data (data), offset (offset), N (N), stride (stride)
#line 43 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
                                                                                                                                       {}
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 36 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 45 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  T & Vec1DView <T>::operator () (size_t i)
#line 46 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    {
        return data.get()[stride*i + offset];
    }
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 36 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 49 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  T Vec1DView <T>::operator () (size_t i) const
#line 50 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    {
        return data.get()[stride*i + offset];
    }
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 36 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 53 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  Vec1DView <T> Vec1DView <T>::slice (size_t start, size_t end)
#line 54 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    {
        return Vec1DView<T>(data, end - start);
    }
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 36 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 57 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  Vec1DView <T> Vec1DView <T>::slice (size_t start)
#line 58 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    {
        return Vec1DView<T>(data, N - start);
    }
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 36 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 61 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  size_t Vec1DView <T>::size () const
#line 62 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    {
        return N;
    }
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 36 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 66 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  Vec1DView <T> Vec1DView <T>::operator += (Vec1DView <T> const & other)
#line 67 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    {
        for (size_t i = 0; i < N; i++)
        {
            data.get()[i + offset] += other(i);
        }
        return *this;
    }
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 36 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 75 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  Vec1DView <T> Vec1DView <T>::operator += (std::vector <T> const & other)
#line 76 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    {
        for (size_t i = 0; i < N; i++)
        {
            data.get()[i + offset] += other[i];
        }
        return *this;
    }
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 36 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 84 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  Vec1DView <T>::operator std::vector <T> ()
#line 85 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    {
        std::vector<T> result(N);
        for (size_t i = 0; i < N; i++)
        {
            result[i] = data.get()[i + offset];
        }
        return result;
    }
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 95 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 101 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  Vec2DView <T>::Vec2DView (std::unique_ptr <T> & data, size_t offset, size_t N0, size_t N1)
#line 101 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    : data (data), offset (offset), N0 (N0), N1 (N1)
#line 101 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
                                                                                                                          {}
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 95 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 103 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  T & Vec2DView <T>::operator () (size_t i, size_t j)
#line 104 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    {
        return data.get()[offset + N0*i + j];
    }
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 95 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 107 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  T Vec2DView <T>::operator () (size_t i, size_t j) const
#line 108 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    {
        return data.get()[offset + N0*i + j];
    }
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 95 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 112 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  Vec1DView <T> Vec2DView <T>::column_view (size_t j)
#line 113 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    {
        return Vec1DView<T>(data, offset + j, N0, N1);
    }
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 95 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 117 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  Vec1DView <T> Vec2DView <T>::operator () (size_t i)
#line 118 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    {
        return Vec1DView<T>(data, offset + i*N0);
    }
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 95 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 122 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  size_t Vec2DView <T>::size () const
#line 123 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    {
        return N0 * N1;
    }
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 95 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 127 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  Vec2DView <T>::operator Vec2D <T> ()
#line 128 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    {
        Vec2D<T> result(N0);
        for (size_t i = 0; i < N0; i++)
        {
            result[i] = Vec1DView<T>(data, offset + i*N1, N1);
        }
        return result;
    }
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 139 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 144 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  LinVec2D <T>::LinVec3D (size_t N0, size_t N1)
#line 144 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    : N0 (N0), N1 (N1)
#line 145 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    {
        data = std::make_unique<T>(N0 * N1);
    }
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 139 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 148 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  T & LinVec2D <T>::operator () (size_t i, size_t j)
#line 149 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    {
        return data.get()[i * N1 + j];
    }
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 139 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 152 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  T LinVec2D <T>::operator () (size_t i, size_t j) const
#line 153 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    {
        return data.get()[i*N1 + j];
    }
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 139 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 156 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  Vec2DView <T> LinVec2D <T>::operator () (size_t row)
#line 157 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    {
        return Vec1DView<T>(data,  N0*row, N1);
    }
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 139 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 160 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  size_t LinVec2D <T>::size () const
#line 161 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    {
        return N0 * N1;
    }
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 139 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 165 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  LinVec2D <T>::operator Vec2D <T> ()
#line 166 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    {
        Vec2D<T> result(N0, std::vector<T>(N1));
        for (size_t i = 0; i < N0; i++)
        {
            result[i] = Vec1DView<T>(data, i*N1, N1);
        }
        return result;
    }
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 177 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 182 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  LinVec3D <T>::LinVec3D (size_t N0, size_t N1, size_t N2)
#line 182 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    : N0 (N0), N1 (N1), N2 (N2)
#line 183 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    {
        data = std::make_unique<T>(N0 * N1 * N2);
    }
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 177 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 186 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  T & LinVec3D <T>::operator () (size_t i, size_t j, size_t k)
#line 187 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    {
        return data.get()[i * N1 * N2 + j * N2 + k];
    }
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 177 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 190 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  T LinVec3D <T>::operator () (size_t i, size_t j, size_t k) const
#line 191 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    {
        return data.get()[i * N1 * N2 + j * N2 + k];
    }
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 177 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 194 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  Vec2DView <T> LinVec3D <T>::operator () (size_t row)
#line 195 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    {
        return Vec2DView<T>(data,  N0*row, N1, N2);
    }
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 177 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 198 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  size_t LinVec3D <T>::size () const
#line 199 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    {
        return N0 * N1 * N2;
    }
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 177 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  template <typename T>
#line 203 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  LinVec3D <T>::operator Vec3D <T> ()
#line 204 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
    {
        Vec3D<T> result(N0);
        for (size_t i = 0; i < N0; i++)
        {
            result[i] = Vec2DView<T>(data, i*N1*N2, N1, N2);
        }
        return result;
    }
}
#undef LZZ_INLINE
#endif
