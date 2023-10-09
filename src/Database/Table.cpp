#include <Sycl_Graph/Database/Table.hpp>
void create_table(pqxx::connection &con, const std::string &table_name,
                             const std::vector<std::string> &indices,
                             const std::vector<std::string> &data_names,
                             std::vector<std::string> data_types)
{
    auto max_constraint = [](std::string idx_name, auto N_max)
    { return "CONSTRAINT max_index_" + idx_name + " CHECK (" + idx_name + " < " + std::to_string(N_max) + ")"; };

    auto N_indices = indices.size();
    if (data_types.empty())
    {
        data_types = std::vector<std::string>(N_indices, "real[]");
    }

    std::stringstream indices_ss;
    for (int i = 0; i < N_indices; i++)
    {
        indices_ss << indices[i] << " INTEGER NOT NULL " << max_constraint(indices[i], N_indices);
        indices_ss << ",";
    }

    std::stringstream data_ss;

    for (int i = 0; i < N_indices; i++)
    {
        data_ss << data_names[i] << " " << data_types[i] << " NOT NULL, ";
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
                          ")";
    work.exec(command.c_str());
    work.commit();
}

std::string string_array(const std::vector<std::string>& strs)
{
    std::stringstream ss;
    for(int i = 0; i < strs.size(); i++)
    {
        ss << strs[i];
        if (i < strs.size() - 1)
        {
            ss << ", ";
        }
    }
    return ss.str();
}
