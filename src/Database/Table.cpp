#include <Sycl_Graph/Database/Table.hpp>
#include <Sycl_Graph/Utils/Validation.hpp>
void create_table(pqxx::connection &con, const std::string &table_name,
                  const std::vector<std::string> &indices,
                  const std::vector<std::string> &data_names,
                  std::vector<std::string> data_types,
                  const std::vector<uint32_t> &max_indices)
{

    if_false_throw(data_types.empty() || data_types.size() == data_names.size(), "data_types.size() != data_names.size()");
    if_false_throw(max_indices.empty() || max_indices.size() == indices.size(), "max_indices.size() != indices.size()");
    auto max_constraint = [](std::string idx_name, auto N_max)
    { return "CONSTRAINT max_index_" + idx_name + " CHECK (" + idx_name + " < " + std::to_string(N_max) + ")"; };

    auto N_indices = indices.size();
    auto N_data_types = data_names.size();
    if (data_types.empty())
    {
        data_types = std::vector<std::string>(N_data_types, "real[]");
    }
    bool is_max_indices = max_indices.size() == N_indices;
    std::stringstream indices_ss;
    std::string constraint_str;
    for (int i = 0; i < N_indices; i++)
    {
        constraint_str = (is_max_indices) ? max_constraint(indices[i], max_indices[i]) : "";
        indices_ss << "\"" << indices[i] << "\""
                   << " INTEGER NOT NULL " << constraint_str;
        indices_ss << ",";
    }

    std::stringstream data_ss;

    for (int i = 0; i < N_data_types; i++)
    {
        data_ss << "\"" << data_names[i] << "\""
                << " " << data_types[i] << " NOT NULL, ";
    }

    std::stringstream pk_ss;
    pk_ss << " PRIMARY KEY (";
    for (int i = 0; i < N_indices; i++)
    {
        if (i < N_indices - 1)
        {
            pk_ss << indices[i] + ", ";
        }
        else
        {
            pk_ss << indices[i];
        }
    }
    pk_ss << ")";

    auto work = pqxx::work(con);
    std::string command = "CREATE TABLE IF NOT EXISTS " + table_name + " (" +
                          indices_ss.str() + data_ss.str() +
                          pk_ss.str() +
                          ");";

    work.exec(command.c_str());
    work.commit();
}

void read_vector(pqxx::row &work_data, const std::string &column_name, uint32_t N_cols, std::vector<float> &result)
{
    auto stof = [](const std::string &str)
    {
        return std::stof(str);
    };
    result = _detail::read_vector_impl<float>(work_data, column_name, stof, N_cols);
}

void read_vector(pqxx::row &work_data, const std::string &column_name, uint32_t N_cols, std::vector<int> &result)
{
    auto stoi = [](const std::string &str)
    { return std::stoi(str); };
    result = _detail::read_vector_impl<int>(work_data, column_name, stoi, N_cols);
}
bool data_type_is_array(const std::string &dtype_str)
{
    // if [] in dtype_str
    return dtype_str.find("[]") != std::string::npos;
}
std::string string_array(const std::vector<std::string> &strs)
{
    std::stringstream ss;
    for (int i = 0; i < strs.size(); i++)
    {
        ss << strs[i];
        if (i < strs.size() - 1)
        {
            ss << ", ";
        }
    }
    return ss.str();
}

void delete_table(pqxx::connection &con, const std::string &table_name)
{
    auto work = pqxx::work(con);
    work.exec("DROP TABLE IF EXISTS " + table_name);
    work.commit();
}

void drop_table(pqxx::connection &con, const std::string &table_name)
{
    auto work = pqxx::work(con);
    work.exec(("DROP TABLE IF EXISTS " + table_name).c_str());
    work.commit();
}
