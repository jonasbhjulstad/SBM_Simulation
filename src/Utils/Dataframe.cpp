#ifndef SYCL_GRAPH_UTILS_DATAFRAME_HPP
#define SYCL_GRAPH_UTILS_DATAFRAME_HPP
#include <Sycl_Graph/Utils/Validation.hpp>
#include <array>
#include <cstdint>
#include <map>
#include <utility>
#include <variant>
#include <algorithm>
#include <type_traits>
#include <vector>
template <typename T, std::size_t N>
struct Dataframe_t;

template <typename T>
struct Dataframe_t<T, 1> : public std::vector<T>
{
    using std::vector<T>::vector;
    Dataframe_t(const std::vector<T>& v) : std::vector<T>(v) {}
    template <typename uI_t>
    Dataframe_t(const std::array<uI_t, 1>& sizes) : std::vector<T>(sizes[0]) {}
    Dataframe_t(std::size_t N): std::vector<T>(N) {}
    Dataframe_t() = default;

    //assignment operator


    template <typename uI_t>
    T operator()(const std::array<uI_t, 1> &&idx) const
    {
        validate_range(idx);
        return this->operator[](idx[0]);
    }

    Dataframe_t<T, 1> operator()(std::array<std::size_t, 1> start, std::array<std::size_t, 1> end) const
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

    void resize_dim(std::size_t dim, std::size_t size)
    {
        if_false_throw(dim == 0, "Specified resize dimension is larger than dataframe dimension");
        this->resize(size);
    }

    template <std::unsigned_integral uI_t>
    void resize_dim(std::size_t dim, const std::vector<uI_t> size)
    {
        if_false_throw(dim == 0, "Specified resize dimension is larger than dataframe dimension");
        this->resize(size[0]);
    }
};

template <typename T, std::size_t N>
auto sub_array(const std::array<T, N>& arr)
{
    std::array<T, N-1> result;
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
    T operator()(uI_t first, auto ... args) const
    {
        return this->operator()(std::array<uI_t, N>{first, args...});
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

    void concatenate(const Dataframe_t<T, N> &df, std::size_t axis)
    {
        if_false_throw(axis < N, "Specified concatenation axis outside of dataframe dimensions");
        if (axis == 1)
        {
            // insert into this
            for (int i = 0; i < this->data.size(); i++)
            {
                this->data[i].insert(this->data[i].begin(), df.data[i].begin(), df.data[i].end());
            }
        }
        else
        {
            for (int i = 0; i < this->data.size(); i++)
            {
                this->data[i].concatenate(df.data[i], axis - 1);
            }
        }
    }

    // template <typename T, std::size_t N, std::size_t N_frames>
    // Dataframe_t<T,N> dataframe_apply(const std::array<Dataframe_t<T,N>, N_frames>& dfs, T (*f)(const Dataframe_t<T,N>&));

    // template <typename T, std::size_t N, std::size_t N_frames>
    // Dataframe_t<T,N> dataframe_apply(const std::array<Dataframe_t<T,N>, N_frames>& dfs, auto f)
    // {
    //     //iterate over all elements in the dataframe
    //     Dataframe_t<T,N> result(dfs[0].get_ranges());
    //     //iterate over ranges
    //     for(int i = 0; i < result.data.size(); i++)
    //     {
    //         std::array<const Dataframe_t<T,N-1>&, N_frames> sub_dfs;
    //         for(int j = 0; j < N_frames; j++)
    //         {
    //             sub_dfs[j] = dfs[j][i];
    //         }
    //         result.data[i] = dataframe_apply(sub_dfs, f);
    //     }
    // }

    Dataframe_t<T, N> operator()(const std::array<std::size_t, N> &&start, const std::array<std::size_t, N> &&end)
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

    template <std::unsigned_integral uI_t>
    void resize_dim(std::size_t dim, const std::vector<uI_t> sizes)
    {
        if_false_throw(dim < N, "Specified resize dimension is larger than dataframe dimension: " + std::to_string(dim) + " vs " + std::to_string(N));
        if (dim == 1)
        {
            if_false_throw(sizes.size() != data.size(), "New sizes for dataframe does not match the dimension size of the dataframe");
                std::for_each(sizes.begin(), sizes.end(), [&, n = 0](auto s) mutable
                              {
                data[n].resize(s);
                n++; });
        }
        else
        {
            auto subsize = std::vector<uI_t>(sizes.begin() + 1, sizes.end());
            if (subsize.size() == 1)
            {
                std::for_each(data.begin(), data.end(), [dim, subsize](auto &d)
                          { d.resize_dim(dim - 1, subsize[0]); });
            }
            else
            {
                std::for_each(data.begin(), data.end(), [dim, subsize](auto &d)
                          { d.resize_dim(dim - 1, subsize); });
            }
        }
    }
};

template <typename First_t, typename ... Ts, std::size_t N>
std::vector<std::tuple<const Dataframe_t<First_t, N-1>&, const Dataframe_t<Ts, N-1>& ...>> dataframe_tie(const Dataframe_t<First_t, N>& df_0, const Dataframe_t<Ts, N>& ... dfs)
{
    std::vector<std::tuple<const Dataframe_t<First_t, N-1>&, const Dataframe_t<Ts, N-1>& ...>> result;
    result.reserve(df_0.size());
    for(int i = 0; i < df_0.size(); i++)
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
        return zip_merge(d0, d1, axis - 1);
    });
    return result;
}







#endif
