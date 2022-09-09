#include <fstream>
#include <DataFrame/DataFrame.h>

// TEST(Regression, single_response_regression)
// {
//     size_t N_features = 2;
//     size_t N_samples = 10;
//     Mat X = Mat::Zero(N_samples, N_features);
//     //Linear feature
//     X.col(0) = Vec::LinSpaced(N_samples, 0, 1);
//     //Quandratic feature
//     X.col(1) = X.col(0).cwiseProduct(X.col(0));

//     Vec y = X.col(1);

//     double ERR_tolerance = 1e-4;
//     auto result = FROLS::Regression::single_response_batch(X, y, ERR_tolerance);

// }

Vec df_col(hmdf::StdDataFrame<double>& df, const char* colname)
{
    using VecMap = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>;
    return VecMap(df.get_column<double>(colname).data(), df.get_column<double>(colname).size());
}

int main()
{
    using namespace hmdf;
    StdDataFrame<double> df;
    df.read("C:\\Users\\jonas\\Documents\\Network_Robust_MPC\\Cpp\\Data\\Bernoulli_SIR_MC_1582.csv", io_format::csv2);


    size_t N_rows = df.get_column<double>("S").size();
    size_t N_features = df.get_columns_info().size();

    std::vector<std::vector<double>> tmp = {df.get_column<double>("S"), df.get_column<double>("I"), df.get_column<double>("R"), df.get_column<double>("p_I")};
    Mat Y(N_rows, N_features);
    Y << df_col(df, "S"), df_col(df, "I"), df_col(df, "R");

    Mat X(N_rows, 4);
    X << Y, df_col(df, "p_I");

    df.write<std::ostream, std::string, double>(std::cout, io_format::csv2);
    FROLS::Regression::multiple_response_batch(X, Y, 1e-4);
    int a = 1;
}