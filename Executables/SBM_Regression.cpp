#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <Sycl_Graph/path_config.hpp>
#include <string>
#include <sstream>
using Mat = Eigen::MatrixXf;
using Vec = Eigen::VectorXf;
using namespace std;
std::string fpath = Sycl_Graph::SYCL_GRAPH_DATA_DIR + std::string("/SIR_sim/");
auto inf_filename = [](uint32_t idx){return std::string("community_infs_" + std::to_string(idx) + ".csv");};
auto connection_inf_filename = [](uint32_t idx){return std::string("connection_infs_" + std::to_string(idx) + ".csv");};
auto rec_filename = [](uint32_t idx){return std::string("community_recs_" + std::to_string(idx) + ".csv");};
auto tot_traj_filename = [](uint32_t idx){return std::string("tot_traj_" + std::to_string(idx) + ".csv");};
static constexpr size_t MAXBUFSIZE=100000;

using namespace Eigen;

MatrixXf openData(string fileToOpen)
{
 
    // the inspiration for creating this function was drawn from here (I did NOT copy and paste the code)
    // https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix
     
    // the input is the file: "fileToOpen.csv":
    // a,b,c
    // d,e,f
    // This function converts input file data into the Eigen matrix format
 
 
 
    // the matrix entries are stored in this variable row-wise. For example if we have the matrix:
    // M=[a b c 
    //    d e f]
    // the entries are stored as matrixEntries=[a,b,c,d,e,f], that is the variable "matrixEntries" is a row vector
    // later on, this vector is mapped into the Eigen matrix format
    vector<float> matrixEntries;
 
    // in this object we store the data from the matrix
    ifstream matrixDataFile(fileToOpen);
 
    // this variable is used to store the row of the matrix that contains commas 
    string matrixRowString;
 
    // this variable is used to store the matrix entry;
    string matrixEntry;
 
    // this variable is used to track the number of rows
    int matrixRowNumber = 0;
 
 
    while (getline(matrixDataFile, matrixRowString)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
    {
        stringstream matrixRowStringStream(matrixRowString); //convert matrixRowString that is a string to a stream variable.
 
        while (getline(matrixRowStringStream, matrixEntry, ',')) // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
        {
            matrixEntries.push_back(stod(matrixEntry));   //here we convert the string to double and fill in the row vector storing all the matrix entries
        }
        matrixRowNumber++; //update the column numbers
    }
 
    // here we convet the vector variable into the matrix and return the resulting object, 
    // note that matrixEntries.data() is the pointer to the first memory location at which the entries of the vector matrixEntries are stored;
    return Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
 
}

Vec get_init_state(std::string filename)
{
    //read first line
    std::ifstream file(filename);
    std::string line;
    std::getline(file, line);
    std::stringstream ss(line);
    std::vector<float> init_state;
    float val;
    while (ss >> val)
    {
        init_state.push_back(val);
        if (ss.peek() == ',')
            ss.ignore();
    }
    return Vec::Map(init_state.data(), init_state.size());
}

int main()
{
    Vec init_state = get_init_state(fpath + tot_traj_filename(0));

    Mat infs = openData(fpath + inf_filename(0));
    Mat recs = openData(fpath + rec_filename(0));
    Mat connection_infs = openData(fpath + connection_inf_filename(0));

    auto N_connections = connection_infs.cols();
    auto Nt = infs.rows();
    auto N_communities = recs.cols();
    std::cout << connection_infs << std::endl;

    

    


    return 0;
}