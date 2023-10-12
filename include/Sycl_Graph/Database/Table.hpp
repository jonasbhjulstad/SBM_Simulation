#ifndef SYCL_GRAPH_DATABASE_TABLE_HPP
#define SYCL_GRAPH_DATABASE_TABLE_HPP
#include <array>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <Sycl_Graph/Utils/String_Manipulation.hpp>
template <typename T>
bool is_string_entry(const T &entry)
{
    return std::get<0>(entry) == pqxx::array_parser::juncture::string_value;
};

void create_table(pqxx::connection &con, const std::string &table_name,
                  const std::vector<std::string> &indices,
                  const std::vector<std::string> &data_names,
                  std::vector<std::string> data_types,
                  const std::vector<uint32_t>& max_indices = {});



std::string string_array(const std::vector<std::string> &strs);

template <typename... Ts>
void table_insert(pqxx::work &work, const std::string &table_name,
                  const std::vector<std::string> &index_names,
                  const std::vector<uint32_t> &index_values,
                  const std::vector<std::string> &data_names,
                  const std::tuple<Ts...> &data)
{
    std::stringstream command;
    // work.exec("INSERT INTO " + table_name + " (p_out, graph, time, state) VALUES (" + std::to_string(p_out) + ", " + std::to_string(graph) + ", " + std::to_string(t) + ", " + state + ")");
    command << "INSERT INTO " << table_name << " (";
    for (int i = 0; i < index_names.size(); i++)
    {
        command << index_names[i];
        if (i < index_names.size() - 1)
        {
            command << ", ";
        }
    }

    for (int i = 0; i < data_names.size(); i++)
    {
        command << ", " << data_names[i];
    }

    command << ") VALUES (";
    for (int i = 0; i < index_values.size(); i++)
    {
        command << std::to_string(index_values[i]) << ", ";
    }
    command << tuple_to_string(data);

    command << ") ON CONFLICT ON CONSTRAINT " << table_name << "_pkey DO UPDATE SET ";

    auto data_str = tuple_to_str_vec(data);
    for (int i = 0; i < data_names.size(); i++)
    {
        command << data_names[i] << " = " << data_str[i];
        if (i < data_names.size() - 1)
        {
            command << ", ";
        }
    }
    command << ";";
    work.exec(command.str().c_str());
}

template <typename... Ts>
void table_insert(pqxx::connection &con, const std::string &table_name,
                  const std::vector<std::string> &index_names,
                  const std::vector<uint32_t> &index_values,
                  const std::vector<std::string> &data_names,
                  const std::tuple<Ts...> &data)
{
    auto work = pqxx::work(con);
    table_insert(work, table_name, index_names, index_values, data_names, data);
    work.commit();
}

bool data_type_is_array(const std::string &dtype_str);

template <std::size_t N>
void parse_data_element(std::array<float, N> &result, auto &data_elem)
{
    auto data_array = data_elem.as_array();
    auto entry = data_array.get_next();
    for (int i = 0; i < result.size(); i++)
    {
        entry = data_array.get_next();
        if (!is_string_entry(entry))
        {
            throw std::runtime_error("Expected more columns database timeseries, got to index: " + std::to_string(i) + " of " + std::to_string(result.size()));
        }
        entry = data_array.get_next();
        result[i] = std::stoi(std::get<1>(entry));
    }
}

template <typename T>
void parse_data_element(T &result, const auto &data_elem)
{
    result = data_elem.template as<T>();
}

namespace _detail
{
    template <typename Tup_t, std::size_t... Is>
    void extract_work_result_impl(Tup_t &datas, const auto &work_res, const auto &data_names, std::index_sequence<Is...>)
    {
        (parse_data_element(std::get<Is>(datas), work_res[data_names[Is]]), ...);
    }
}
template <typename... Ts>
void extract_work_result(std::tuple<Ts...> &datas, const auto &work_res, const auto &data_names)
{
    constexpr auto index_seq = std::make_index_sequence<sizeof...(Ts)>{};
    _detail::extract_work_result_impl(datas, work_res, data_names, index_seq);
}

template <typename... Ts>
std::tuple<Ts...> read_table_column(pqxx::connection &con, const std::string &table_name,
                                    const std::vector<std::string> &index_names,
                                    const std::vector<uint32_t> &index_values,
                                    const std::vector<std::string> &data_names,
                                    const std::vector<std::string> &data_types)
{
    auto work = pqxx::work(con);
    auto data_fields = string_array(data_names);

    auto command = "SELECT " + data_fields + " FROM " + table_name + " WHERE ";
    for (int i = 0; i < index_names.size(); i++)
    {
        command += index_names[i] + " = " + std::to_string(index_values[i]);
        if (i < index_names.size() - 1)
        {
            command += " AND ";
        }
    }

    auto data = work.exec1(command);
    std::tuple<Ts...> result{};
    // array_parser
    extract_work_result(result, data, data_names);
    return result;
}
namespace _detail
{
    //define alias for functions with same type as std::stof
    template <typename T>
    std::vector<T> read_vector_impl(pqxx::row &work_data, const std::string &column_name, T (*f)(const std::string&), std::size_t N_cols)
    {
        auto w_array = work_data[column_name.c_str()].as_array();
        auto entry = w_array.get_next();
        std::vector<T> result(N_cols);
        for (int j = 0; j < N_cols; j++)
        {
            entry = w_array.get_next();
            if (!is_string_entry(entry))
            {
                throw std::runtime_error("Expected more columns database timeseries, got to index: " + std::to_string(j) + " of " + std::to_string(N_cols));
            }
            result[j] = f(std::get<1>(entry));
        }
        return result;
    }
}

void read_vector(pqxx::row &work_data, const std::string &column_name, uint32_t N_cols, std::vector<float>& result);

void read_vector(pqxx::row &work_data, const std::string &column_name, uint32_t N_cols, std::vector<int>& result);

template <typename... Ts>
std::vector<std::tuple<Ts...>> read_table_slice(pqxx::connection &con, const std::string &table_name,
                                   const std::vector<std::string> &fixed_index_names,
                                   const std::vector<uint32_t> &fixed_index_values,
                                   const std::string& slice_index_name,
                                   std::pair<uint32_t, uint32_t> slice_index_value,
                                   const std::vector<std::string> &data_names,
                                   const std::vector<std::string> &data_types)
{
    auto N_rows = slice_index_value.second - slice_index_value.first;

    auto work = pqxx::work(con);
    auto data_fields = string_array(data_names);

    auto command = "SELECT " + data_fields + " FROM " + table_name + " WHERE ";
    for (int i = 0; i < fixed_index_names.size(); i++)
    {
        command += fixed_index_names[i] + " = " + std::to_string(fixed_index_values[i]) + " AND ";
    }
    command += slice_index_name + " >= " + std::to_string(slice_index_value.first) + " AND " + slice_index_name + " < " + std::to_string(slice_index_value.second);

    auto data = work.exec_n(N_rows, command);
    std::vector<std::tuple<Ts...>> result(N_rows, std::tuple<Ts ...>{});
    for(int i = 0; i < N_rows; i++)
    {
        extract_work_result(result[i], data[i], data_names);
    }

    return result;
}
void delete_table(pqxx::connection &con, const std::string &table_name);

void drop_table(pqxx::connection &con, const std::string& table_name);

#endif
