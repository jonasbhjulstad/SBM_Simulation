#ifndef SYCL_GRAPH_UTILS_DATAFRAME_HPP
#define SYCL_GRAPH_UTILS_DATAFRAME_HPP
#include <array>
#include <cstdint>
#include <utility>
#include <vector>
template <typename T, std::size_t N>
struct Dataframe_t;

template <typename T>
struct Dataframe_t<T, 1> : public std::vector<T>
{
    Dataframe_t<T, 1> operator()(std::array<std::size_t, 1> start, std::array<std::size_t, 1> end)
    {
        return Dataframe_t<T, 1>(this->begin() + start[0], this->begin() + end[0]);
    }
};

template <typename T, std::size_t N>
struct Dataframe_t
{
    // N arguments
    Dataframe_t(std::size_t size_0, std::size_t... sizes) : data(size_0, Dataframe_t<T, N - 1>(sizes...)) {}
    std::vector<Dataframe_t<T, N - 1>> data;
    std::size_t size() const { return data.size(); }
    Dataframe_t<T, N - 1> &operator[](std::size_t idx) { return data[idx]; }
    std::array<std::size_t, N> shape() const
    {
        std::array<std::size_t, N> s;
        s[0] = data.size();
        auto sub_shape = data[0].shape();
        std::copy(sub_shape.begin(), sub_shape.end(), s.begin() + 1);
        return s;
    }
    std::size_t get_range(const std::vector<std::size_t> &&idx) const
    {
        if (idx.size() == 1)
        {
            return data[idx[0]].size();
        }
        else
        {
            return data[idx[0]].get_range(std::forward < const std::vector<std::size_t>(std::vector<std::size_t>(idx.begin() + 1, idx.end())));
        }
    }

    void validate_range(const std::array<std::size_t, N> &range) const
    {
        if (range[0] != data.size())
        {
            throw std::runtime_error("Invalid range");
        }
        data[0].validate_range(std::array<std::size_t, N - 1>(range.begin() + 1, range.end()));
    }

    T operator()(const std::array<std::size_t, N> &&idx)
    {
        validate_range(idx);
        return data[idx[0]](std::array<std::size_t, N - 1>(idx.begin() + 1, idx.end()));
    }

    T operator()(std::size_t ... idx)
    {
        return operator()(std::array<std::size_t, N>{idx...});
    }

    Dataframe_t<T, N> operator()(const std::array<std::size_t, N> &start, const std::array<std::size_t, N> &end)
    {
        validate_range(start);
        validate_range(end);
        Dataframe_t<T, N> sliced_df(end - start);
        for (std::size_t i = 0; i < sliced_df.size(); i++)
        {
            sliced_df[i] = data[i](start, end);
        }
        return sliced_df;
    }

    void resize(const Dataframe_t<std::size_t, N-1>& sizes)
    {
        data.resize(sizes.size());
        for(std::size_t i = 0; i < sizes.size(); i++)
        {
            data[i].resize(sizes[i]);
        }
    }
};

#endif
