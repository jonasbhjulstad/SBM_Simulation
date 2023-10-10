#ifndef SYCL_GRAPH_UTILS_DATAFRAME_HPP
#define SYCL_GRAPH_UTILS_DATAFRAME_HPP
#include <vector>
#include <array>
#include <cstdint>
#include <tuple>
#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <Sycl_Graph/Utils/Validation.hpp>
// #include <Sycl_Graph/Epidemiological/SIR_Types.hpp>
template <typename T, std::size_t N>
struct Dataframe_t;

template <typename T>
struct Dataframe_t<T, 1> : public std::vector<T>
{
    using std::vector<T>::vector;
    Dataframe_t(const std::vector<T> &v) : std::vector<T>(v) {}
    template <typename uI_t>
    Dataframe_t(const std::array<uI_t, 1> &sizes) : std::vector<T>(sizes[0]) {}
    Dataframe_t(std::size_t N) : std::vector<T>(N) {}
    Dataframe_t(const std::array<std::size_t, 1> &sizes) : std::vector<T>(sizes[0]) {}
    Dataframe_t(const std::array<std::size_t, 1>&& sizes) : std::vector<T>(sizes[0]) {}
    Dataframe_t() = default;

    //copy assignment operator
    Dataframe_t(const Dataframe_t<T, 1> &df) : std::vector<T>(df) {}


    // assignment operator

    template <typename uI_t>
    T& at(const std::array<uI_t, 1> &&idx) const
    {
        validate_range(idx);
        return this->operator[](idx[0]);
    }

    template <typename uI_t>
    const T operator()(const std::array<uI_t, 1> &&idx) const
    {
        validate_range(idx);
        return this->operator[](idx[0]);
    }



    const T operator()(std::size_t idx) const
    {
        return this->operator[](idx);
    }


    T& at(std::size_t idx)
    {
        return this->operator[](idx);
    }



    Dataframe_t<T, 1> slice(std::size_t start, std::size_t end) const
    {
        return Dataframe_t<T, 1>(this->begin() + start, this->begin() + end);
    }

    // Dataframe_t<T, 1>& at(std::array<std::size_t, 1> start, std::array<std::size_t, 1> end)
    // {
    //     return Dataframe_t<T, 1>(this->begin() + start[0], this->begin() + end[0]);
    // }

    const Dataframe_t<T, 1> operator()(std::array<std::size_t, 1> start, std::array<std::size_t, 1> end) const
    {
        return Dataframe_t<T, 1>(this->begin() + start[0], this->begin() + end[0]);
    }

    auto apply(auto f) const
    {
        std::vector<decltype(f(std::declval<T>()))> result(this->size());
        std::transform(this->begin(), this->end(), result.begin(), f);
        return result;
    }

    bool all_of(auto f) const
    {
        return std::all_of(this->begin(), this->end(), f);
    }

    std::array<std::size_t, 1> get_ranges() const
    {
        return std::array<std::size_t, 1>{this->size()};
    }

    template <typename uI_t>
    void validate_range(const std::array<uI_t, 1> &range) const
    {
        if (range[0] > this->size())
        {
            throw std::runtime_error("Invalid range");
        }
    }

    template <typename uI_t>
    std::size_t get_range(const std::vector<uI_t>&& idx) const
    {
        throw std::runtime_error("Invalid range");
    }

    void resize_dim(std::size_t dim, std::size_t size)
    {
        if_false_throw(dim == 0, "Specified resize dimension is larger than dataframe dimension");
        this->resize(size);
    }

    template <std::size_t dim>
    auto apply(auto f) const
    {
        static_assert(dim == 1, "Invalid dimension");
        Dataframe_t<decltype(f(std::declval<T>())), dim> result(this->get_ranges());
        std::transform(this->data.begin(), this->data.end(), result.data.begin(), f);
    }

    std::vector<T> flatten() const
    {
        return *this;
    }

    void concatenate(const Dataframe_t<T, 1> &df, std::size_t axis)
    {
        throw std::runtime_error("Invalid axis");
    }



    template <std::unsigned_integral uI_t>
    void resize_dim(std::size_t dim, const std::vector<uI_t> size)
    {
        if_false_throw(dim == 0, "Specified resize dimension is larger than dataframe dimension");
        this->resize(size[0]);
    }

    std::size_t byte_size() const
    {
        return this->size() * sizeof(T);
    }
};

template <typename T, std::size_t N>
auto sub_array(const std::array<T, N> &arr)
{
    std::array<T, N - 1> result;
    std::copy(arr.begin() + 1, arr.end(), result.begin());
    return result;
}

template <typename T, std::size_t N>
struct Dataframe_t
{
    // N arguments
    Dataframe_t() = default;
    template <typename uI_t>
    Dataframe_t(std::array<uI_t, N> sizes) : data(sizes[0], Dataframe_t<T, N - 1>(sub_array(sizes))) {}
    Dataframe_t(std::size_t size_0, auto... sizes) : data(size_0, Dataframe_t<T, N - 1>(sizes...)) {}
    Dataframe_t(const std::vector<Dataframe_t<T, N - 1>> &data) : data(data) {}
    Dataframe_t(std::vector<Dataframe_t<T, N - 1>> &&data) : data(std::move(data)) {}

    template <typename D>
    Dataframe_t(const std::vector<D>& data): data(data.size())
    {
        std::transform(data.begin(), data.end(), this->data.begin(), [](auto& d)
        {
            return Dataframe_t<T, N-1>(d);
        });
    }

    Dataframe_t(const std::vector<Dataframe_t<T, N-1>>::iterator& begin, const std::vector<Dataframe_t<T, N-1>>::iterator& end): data(begin, end) {}
    //default copy assignment operator
    Dataframe_t(const Dataframe_t<T, N> &df) = default;

    static constexpr std::size_t N_dims = N;
    std::vector<Dataframe_t<T, N - 1>> data;
    std::size_t size() const { return data.size(); }

    Dataframe_t<T, N - 1> &operator[](std::integral auto idx) { return data[idx]; }
    const Dataframe_t<T, N - 1> &operator[](std::integral auto idx) const { return data[idx]; }
    std::array<std::size_t, N> get_ranges() const
    {
        std::array<std::size_t, N> s;
        s[0] = data.size();
        auto sub_shape = data[0].get_ranges();
        std::copy(sub_shape.begin(), sub_shape.end(), s.begin() + 1);
        return s;
    }
    std::size_t get_range(const std::vector<std::size_t> idx) const
    {
        if (idx.size() == 1)
        {
            return data[idx[0]].size();
        }
        else
        {
            return data[idx[0]].get_range(std::vector(idx.begin() + 1, idx.end()));
        }
    }

    template <typename uI_t>
    void validate_range(const std::array<uI_t, N> &range) const
    {
        if (range[0] > data.size())
        {
            throw std::runtime_error("Invalid range");
        }
        data[0].validate_range(sub_array(range));
    }

    template <typename uI_t>
    T operator()(const std::array<uI_t, N> &&idx) const
    {
        validate_range(idx);
        return data[idx[0]](sub_array(idx));
    }

    template <typename uI_t>
    T& at(const std::array<uI_t, N> &&idx)
    {
        validate_range(idx);
        return data[idx[0]](sub_array(idx));
    }

    Dataframe_t<T,N> slice(std::size_t start, std::size_t end) const
    {
        Dataframe_t<T, N> result(end - start);
        std::copy(this->begin() + start, this->begin() + end, result.begin());
        return result;
    }

    bool all_of(auto f) const
    {
        return std::all_of(data.begin(), data.end(), [f](auto &d)
                           { return d.all_of(f); });
    }

    Dataframe_t<T, N> operator+(const Dataframe_t<T, N> &df)
    {
        this->data.insert(data.begin(), df.data.begin(), df.data.end());
        return *this;
    }

    void insert(const Dataframe_t<T, N>& df, uint32_t offset = 0)
    {
        this->data.insert(data.begin() + offset, df.data.begin(), df.data.end());
    }

    // void concatenate(const Dataframe_t<T, N> &df, std::size_t axis)
    // {
    //     if_false_throw(axis < N, "Specified concatenation axis outside of dataframe dimensions");
    //     if (axis == 1)
    //     {
    //         // insert into this
    //         for (int i = 0; i < this->data.size(); i++)
    //         {
    //             insert(df.data[i]);
    //         }
    //     }
    //     else
    //     {
    //         for (int i = 0; i < this->data.size(); i++)
    //         {
    //             this->data[i].concatenate(df.data[i], axis - 1);
    //         }
    //     }
    // }

    T operator()(std::size_t first, auto&& ... rest) const
    {
        return this->operator()(std::array<std::size_t, N>{first, (std::size_t)rest...});
    }


    Dataframe_t<T, N> slice(std::size_t start, std::size_t end)
    {
        return Dataframe_t<T, N>(this->begin() + start, this->begin() + end);
    }

    Dataframe_t<T, N> operator()(const std::array<std::size_t, N> &&start, const std::array<std::size_t, N> &&end) const
    {
        auto array_diff = [](const std::array<std::size_t, N> &a0, const std::array<std::size_t, N> &a1)
        {
            std::array<std::size_t, N> result;
            std::transform(a0.begin(), a0.end(), a1.begin(), result.begin(), [](auto a0, auto a1)
                           { return a0 - a1; });
            return result;
        };

        validate_range(start);
        validate_range(end);
        Dataframe_t<T, N> sliced_df(array_diff(end, start));
        for (std::size_t i = 0; i < sliced_df.size(); i++)
        {
            sliced_df[i] = data[i](sub_array(start), sub_array(end));
        }
        return sliced_df;
    }

    auto apply(auto f) const
    {
        Dataframe_t<decltype(f(std::declval<T>())), N> result(this->get_ranges());
        std::transform(this->data.begin(), this->data.end(), result.data.begin(), [f](auto &d)
                       { return d.apply(f); });
        return result;
    }

    template <std::size_t dim, typename Ret_t>
    auto apply(auto f) const
    {
        //decltype(f(std::declval<T>()))
        Dataframe_t<Ret_t, dim> result(this->size());
        if constexpr (dim == 1)
        {
            std::transform(this->cbegin(), this->cend(), result.begin(), f);
        }
        else
        {
            std::transform(this->cbegin(), this->cend(), result.begin(), [f](auto &d)
                        { return d.template apply<dim-1>(f); });
        }
        return result;
    }

    void resize(std::size_t size)
    {
        data.resize(size);
    }

    void resize(const Dataframe_t<std::size_t, N - 1> &sizes)
    {
        data.resize(sizes.size());
        for (std::size_t i = 0; i < sizes.size(); i++)
        {
            data[i].resize(sizes[i]);
        }
    }

    void resize_dim(std::size_t dim, std::size_t size)
    {
        if (dim == 0)
        {
            data.resize(size);
        }
        else
        {
            std::for_each(data.begin(), data.end(), [dim, size](auto &d)
                          { d.resize_dim(dim - 1, size); });
        }
    }

    // Dataframe_t<T, N - 1> flatten_dim_0() const
    // {
    //     std::vector<uint32_t> size_vec(data.size()+1, 0);
    //     std::transform(data.begin(), data.end(), size_vec.begin()+1, [](auto &d)
    //                    { return d.size(); });
    //     auto tot_size = std::accumulate(size_vec.begin(), size_vec.end(), 1, std::multiplies<std::size_t>());

    //     Dataframe_t<T, N - 1> result(tot_size);
    //     for (int i = 0; i < data.size(); i++)
    //     {
    //         result.insert(result.begin() + size_vec[i], data[i].begin(), data[i].end());
    //     }
    //     return result;
    // }

    std::vector<T> flatten() const
    {
        std::vector<T> result;
        for (auto &d : data)
        {
            auto flattened = d.flatten();
            result.insert(result.end(), flattened.begin(), flattened.end());
        }
        return result;
    }

    std::size_t byte_size() const
    {
        std::size_t result = 0;
        for (auto &d : data)
        {
            result += d.byte_size();
        }
        return result;
    }

    auto cbegin() const
    {
        return data.begin();
    }
    auto cend() const
    {
        return data.cend();
    }

    auto begin()
    {
        return data.begin();
    }
    auto end()
    {
        return data.end();
    }


    auto begin() const
    {
        return data.begin();
    }
    auto end() const
    {
        return data.end();
    }



};

template <typename First_t, typename... Ts, std::size_t N>
std::vector<std::tuple<const Dataframe_t<First_t, N - 1> &, const Dataframe_t<Ts, N - 1> &...>> dataframe_tie(const Dataframe_t<First_t, N> &df_0, const Dataframe_t<Ts, N> &...dfs)
{
    std::vector<std::tuple<const Dataframe_t<First_t, N - 1> &, const Dataframe_t<Ts, N - 1> &...>> result;
    result.reserve(df_0.size());
    for (int i = 0; i < df_0.size(); i++)
    {
        result.push_back(std::tie(df_0[i], dfs[i]...));
    }
    return result;
}

template <typename T>
auto zip_merge(const std::vector<T> &v0, const std::vector<T> &v1, auto dummy_arg = 0)
{
    std::vector<T> result(v0.size() + v1.size());
    for (int i = 0; i < v0.size(); i++)
    {
        result[2 * i] = v0[i];
        result[2 * i + 1] = v1[i];
    }
    return result;
}

template <typename T, std::size_t N>
std::enable_if_t<(N > 1), Dataframe_t<T, N>> zip_merge(const Dataframe_t<T, N> &df_0, const Dataframe_t<T, N> &df_1, std::size_t axis)
{
    Dataframe_t<T, N> result(df_0.get_ranges());
    if_false_throw(axis < N, "Specified concatenation axis outside of dataframe dimensions");
    auto dfs = dataframe_tie(df_0, df_1);
    std::transform(dfs.begin(), dfs.end(), result.data.begin(), [axis](const auto &df_pack)
                   {
        auto [d0, d1] = df_pack;
        return zip_merge(d0, d1, axis - 1); });
    return result;
}

#endif
