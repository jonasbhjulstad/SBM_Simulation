#include <Sycl_Graph/Database/Enum.hpp>
void define_enum(pqxx::connection& con, const std::string& enum_name, const std::vector<std::string>& enum_values)
{
    auto work = pqxx::work(con);
    work.exec("DROP TYPE IF EXISTS " + enum_name + ";");

    std::string enum_str = "CREATE TYPE " + enum_name + " AS ENUM (";
    for (int i = 0; i < enum_values.size(); i++)
    {
        enum_str += "'" + enum_values[i] + "'";
        if (i < enum_values.size() - 1)
        {
            enum_str += ", ";
        }
    }
    enum_str += ");";
    work.exec(enum_str);
    work.commit();
}
