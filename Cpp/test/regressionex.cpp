#include <fstream>
#include <DataFrame/DataFrame.h>

// TEST(Regression, single_response_regression)
// {
//     uint32_t N_features = 2;
//     uint32_t N_samples = 10;
//     Mat X = Mat::Zero(N_samples, N_features);
//     //Linear feature
//     X.col(0) = Vec::LinSpaced(N_samples, 0, 1);
//     //Quandratic feature
//     X.col(1) = X.col(0).cwiseProduct(X.col(0));

//     Vec y = X.col(1);

//     float ERR_tolerance = 1e-4;
//     auto result = FROLS::Regression::single_response_batch(X, y, ERR_tolerance);

// }

Vec df_col(hmdf::StdDataFrame<float>& df, const char* colname)
{
    using VecMap = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>;
    return VecMap(df.get_column<float>(colname).data(), df.get_column<float>(colname).size());
}

int main()
{
    using namespace hmdf;
    StdDataFrame<float> df;
    df.read("C:\\Users\\jonas\\Documents\\Network_Robust_MPC\\Cpp\\Data\\Bernoulli_SIR_MC_1582.csv", io_format::csv2);


    uint32_t N_rows = df.get_column<float>("S").size();
    uint32_t N_features = df.get_columns_info().size();

    std::vector<std::vector<float>> tmp = {df.get_column<float>("S"), df.get_column<float>("I"), df.get_column<float>("R"), df.get_column<float>("p_I")};
    Mat Y(N_rows, N_features);
    Y << df_col(df, "S"), df_col(df, "I"), df_col(df, "R");

    Mat X(N_rows, 4);
    X << Y, df_col(df, "p_I");

    df.write<std::ostream, std::string, float>(std::cout, io_format::csv2);
    FROLS::Regression::multiple_response_batch(X, Y, 1e-4);
    int a = 1;
}