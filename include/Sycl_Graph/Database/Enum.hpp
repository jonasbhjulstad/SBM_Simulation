#ifndef SYCL_GRAPH_DATABASE_ENUM_HPP
#define SYCL_GRAPH_DATABASE_ENUM_HPP
#include <vector>
#include <string>

enum class Community_Type
{
    Detected_Communities,
    True_Communities
};
enum class Regression_Type
{
    LS,
    QR
};

std::string to_string(Community_Type c_type)
{
    switch (c_type)
    {
    case Community_Type::Detected_Communities:
        return "Detected_Communities";
    case Community_Type::True_Communities:
        return "True_Communities";
    }
}

std::string to_string(Regression_Type r_type)
{
    switch (r_type)
    {
    case Regression_Type::LS:
        return "LS";
    case Regression_Type::QR:
        return "QR";
    }
}
void define_enum(pqxx::connection& con, const std::string& enum_name, const std::vector<std::string>& enum_values);

#endif
