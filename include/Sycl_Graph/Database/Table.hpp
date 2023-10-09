#ifndef SYCL_GRAPH_DATABASE_TABLE_HPP
#define SYCL_GRAPH_DATABASE_TABLE_HPP
#include <array>
#include <cstdint>
#include <fstream>
#include <pqxx/pqxx>
#include <sstream>
#include <string>
#include <vector>

template <typename T>
bool is_string_entry(const T &entry)
{
    return std::get<0>(entry) == pqxx::array_parser::juncture::string_value;
};
void create_table(pqxx::connection &con, const std::string &table_name,
                  const std::vector<std::string> &indices,
                  const std::vector<std::string> &data_names,
                  std::vector<std::string> data_types);

template <typename... Ts>
std::string tuple_to_string(const std::tuple<Ts...> &data)
{
    std::stringstream ss;
    std::apply([&ss](auto &&...args)
               { ((ss << args << ", "), ...); },
               data);
    // remove last comma
    auto str = ss.str();
    str.pop_back();
    str.pop_back();
    return str;
}

template <typename... Ts>
std::vector<std::string> tuple_to_str_vec(const std::tuple<Ts...> &data)
{
    std::vector<std::string> vec;
    std::apply([&vec](auto &&...args)
               { ((vec.push_back(std::to_string(args))), ...); },
               data);
    return vec;
}

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

bool data_type_is_array(const std::string& dtype_str)
{
    //if [] in dtype_str
    return dtype_str.find("[]") != std::string::npos;
}

template <std::size_t N>
void parse_data_element(std::array<float, N>& result,  auto& data_elem)
{
    auto data_array = data_elem.as_array();
    auto entry = res.get_next();
    for(int i = 0; i < result.size(); i++)
    {
        entry = res.get_next();
        if (!is_string_entry(entry))
        {
            throw std::runtime_error("Expected more columns database timeseries, got to index: " + std::to_string(j) + " of " + std::to_string(N_cols));
        }
        entry = res.get_next();
        result[i] = std::stoi(std::get<1>(entry));
    }
    return result;
}

template <typename T>
void parse_data_element(T& result, auto& data_elem)
{
    return data_elem.as<T>();
}


template <typename... Ts>
std::tuple<Ts...> read_table(pqxx::connection &con, const std::string &table_name,
                             const std::vector<std::string> &index_names,
                             const std::vector<uint32_t> &index_values,
                             const std::vector<std::string> &data_names,
                             const std::vector<std::string>& data_types)
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

    auto result = work.exec1(command);
    std::tuple<Ts...> data;
    // array_parser
    for (int i = 0; i < result.size(); i++)
    {
        std::apply([&result, &i](auto &&...args)
                   { ((args = parse_data_element(args, result[i][data_names[i]])), ...); },
                   data);
    }

    return data;
}

#endif
