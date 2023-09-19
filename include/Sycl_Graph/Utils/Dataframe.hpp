#ifndef SYCL_GRAPH_UTILS_DATAFRAME_HPP
#define SYCL_GRAPH_UTILS_DATAFRAME_HPP
#include <array>
#include <cstdint>
#include <utility>
#include <vector>
#include <Sycl_Graph/Utils/Validation.hpp>
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

template <typename T>
auto zip_merge_vectors(const std::vector<T> &v0, const std::vector<T> &v1)
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
struct Dataframe_t
{
    // N arguments
    Dataframe_t(std::size_t size_0, std::size_t... sizes) : data(size_0, Dataframe_t<T, N - 1>(sizes...)) {}
    std::vector<Dataframe_t<T, N - 1>> data;
    std::size_t size() const { return data.size(); }
    Dataframe_t<T, N - 1> &operator[](std::size_t idx) { return data[idx]; }
    std::array<std::size_t, N> get_ranges() const
    {
        std::array<std::size_t, N> s;
        s[0] = data.size();
        auto sub_shape = data[0].get_ranges();
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

    T operator()(std::size_t... idx)
    {
        return operator()(std::array<std::size_t, N>{idx...});
    }

    Dataframe_t<T, N> operator+(const Dataframe_t<T,N>& df)
    {
        this->data.insert(data.begin(), df.data.begin(), df.data.end());
        return *this;
    }

    void concatenate(const Dataframe_t<T,N>& df, std::size_t axis)
    {
        if_false_throw(axis < N, "Specified concatenation axis outside of dataframe dimensions");
        if(axis == 1)
        {
            //insert into this
            for(int i = 0; i< this->data.size(); i++)
            {
                this->data[i].insert(this->data[i].begin(), df.data[i].begin(), df.data[i].end());
            }
        }
        else
        {
            for(int i = 0; i < this->data.size(); i++)
            {
                this->data[i].concatenate(df.data[i], axis-1);
            }
        }
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

    void resize_dim(std::size_t dim, const std::vector<std::size_t> sizes)
    {
        if_false_throw(dim < N, "Specified resize dimension is larger than dataframe dimension: " + std::to_string(dim) + " vs " + std::to_string(N));
        if (dim == 1)
        {
            if_false_throw(sizes.size() != data.size(), "New sizes for dataframe does not match the dimension size of the dataframe")
            std::for_each(sizes.begin(), sizes.end(), [&, n = 0](auto s) mutable
                          {
                data[n].resize(s);
                n++; });
        }
        else
        {
            auto subsize = std::vector<std::size_t>(sizes.begin() + 1, sizes.end());
            std::for_each(data.begin(), data.end(), [dim, subsizes](auto &d)
                          { d.resize_dim(dim - 1, subsizes); });
        }
    }
};

template <typename T, std::size_t N>
Dataframe_t<T,N> zip_merge(const Dataframe_t<T,N>& df_0, const Dataframe_t<T,N>& df_1, std::size_t axis)
{
    Dataframe_t<T,N> result(df_0.get_ranges());
    if_false_throw(axis < N, "Specified concatenation axis outside of dataframe dimensions");
    if(axis == 1)
    {
        for(int i = 0; i < df_0->data.size(); i++)
        {
            auto size_prev = this->data[i].size();
            auto df_size = df_0.data[i].size();
            df_0->data[i].resize(size_prev + df_size);
        }
        std::transform(df_0->data.begin(), df_0->data.end(), df_1.data.begin(), result->data.begin(), [](const auto& this_data, const auto& df_data)
        {
            return zip_merge_vectors(this_data, df_data);
        });
        return result;
    }
    else
    {
        for(int i = 0; i < df_0.size(); i++)
        {
            zip_merge(df_0[i], df_1[i], axis-1);
        }
    }
}

#endif
