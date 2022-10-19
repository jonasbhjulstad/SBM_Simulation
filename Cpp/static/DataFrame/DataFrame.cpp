#include "DataFrame.hpp"
#include <FROLS_Path_Config.hpp>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <sys/stat.h>
#include <filesystem>

namespace FROLS {

    void DataFrame::assign(const std::string &col_name, crVec &vec) {
        assign(col_name, std::vector<float>(vec.data(), vec.data() + vec.rows()));
    }

    void DataFrame::assign(const std::vector<std::string> &col_names, crMat &mat) {
        for (int i = 0; i < mat.cols(); i++) {
            assign(col_names[i], mat.col(i));
        }
    }


    void DataFrame::assign(const std::string &col_name,
                           const std::vector<uint32_t> &col_data) {

        data[col_name] = std::make_shared<std::vector<float>>(col_data.size());
        for (int i = 0; i < col_data.size(); i++) {
            (*data[col_name])[i] = col_data[i];
        }

    }

    void DataFrame::assign(const std::string &col_name,
                           const std::vector<float> &col_data) {
        data[col_name] = std::make_shared<std::vector<float>>(col_data);
    }

    void DataFrame::assign(const std::string &col_name, uint32_t row, float value) {
        if (data[col_name]->size() < row+1)
        {
            data[col_name]->resize(row+1);
        }
        data[col_name]->operator[](row) = value;
    }
    const std::vector<std::vector<float>> DataFrame::operator[](const std::vector<std::string>& col_names) const
    {
        std::vector<std::vector<float>> res(col_names.size());
        std::transform(col_names.begin(), col_names.end(), res.begin(), [&](const std::string& name)
        {
            return *this->operator[](name);
        });
        return res;
    }
    void DataFrame::assign(const std::vector<std::string> &col_names,
                           const std::vector<std::vector<uint32_t>> &col_data) {
        for (uint32_t i = 0; i < col_names.size(); i++) {
            std::vector<float> col(col_data[i].size());
            std::copy(col_data[i].begin(), col_data[i].end(), col.begin());
            data[col_names[i]] = std::make_shared<std::vector<float>>(col);
        }
    }

    void DataFrame::assign(const std::vector<std::string> &col_names,
                           const std::vector<std::vector<float>> &col_data) {
        for (uint32_t i = 0; i < col_names.size(); i++) {
            *data[col_names[i]] = col_data[i];
        }
    }

    std::vector<std::string> DataFrame::get_col_names() {
        std::vector<std::string> col_names;
        for (auto &col: data) {
            col_names.push_back(col.first);
        }
        return col_names;
    }

    void DataFrame::read_csv(const std::string &filename, const std::string &delimiter) {
        std::ifstream file(filename);
        std::string line;
        std::getline(file, line);
        std::stringstream ss(line);
        std::string col_name;
        std::vector<std::string> col_names;
        while (std::getline(ss, col_name, delimiter[0])) {
            col_names.push_back(col_name);
        }
        for (auto &col_name: col_names) {
            std::shared_ptr<std::vector<float>> ptr =
                    std::make_shared<std::vector<float>>(N_rows);
            data.insert({col_name, ptr});
        }
        uint32_t row = 0;
        N_rows = 0;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string value;
            uint32_t col = 0;
            while (std::getline(ss, value, delimiter[0])) {
                this->assign(col_names[col], row, std::stod(value));
                col++;
                if (col > N_rows) {
                    N_rows += 100;
                    for (auto &col_name: col_names) {
                        this->data[col_name]->resize(N_rows);
                    }
                }
            }
            row++;
        }
        N_rows = row;
        for (auto &col_name: col_names) {
            this->data[col_name]->resize(N_rows);
        }
    }

    void DataFrame::write_csv(const std::string &filename,
                              const std::string &delimiter,
                              const float termination_tol) {
        std::string filedir = FROLS::path_dirname(filename);
        if (!std::filesystem::is_directory(filedir))
        {
            std::filesystem::create_directories(FROLS::path_dirname(filename));
        }

        std::ofstream file(filename);
        uint32_t N_cols = data.size();
        uint32_t iter = 0;
        for (auto &col: data) {
            iter++;
            file << col.first;
            if (iter < N_cols) {
                file << delimiter;
            }
        }
        file << "\n";
        for (uint32_t i = 0; i < N_rows; i++) {
            std::vector<float> coldata(data.size());
            std::transform(data.begin(), data.end(), coldata.begin(),
                           [&i](auto &col) { return col.second->operator[](i); });
            if (std::all_of(coldata.begin(), coldata.end(), [&](auto &cd) { return cd < termination_tol; })) {
                break;
            }
            std::for_each(coldata.begin(), coldata.end(), [&, n = 1](auto &d) mutable { file << d;
                file << ((n != coldata.size()) ? "," : "\n");
            n++;});
        }
    }

    void DataFrame::resize(uint32_t N_rows) {
        if (!data.empty()) {
            for (auto &col: data) {
                col.second->resize(N_rows);
            }
        }
        this->N_rows = N_rows;
    }

    void DataFrame::resize() {
        uint32_t rows = 0;
        for (const auto &col: data) {
            rows = std::max({(size_t)rows, col.second->size()});
        }
        N_rows = rows;
    }


    const std::vector<float> DataFrameStack::get_col(uint32_t frame, const std::string &col_name) {
        assert(frame < dataframes.size());
        return *(dataframes[frame][col_name]);
    }

    std::vector<float> DataFrameStack::get_row(uint32_t frame, uint32_t row) {
        std::vector<float> row_data;
        for (auto &col: dataframes[frame].data) {
            row_data.push_back(col.second->operator[](row));
        }
        return row_data;
    }

    std::vector<uint32_t> DataFrameStack::get_N_rows() {
        std::vector<uint32_t> N_rows(dataframes.size());
        std::generate(N_rows.begin(), N_rows.end(), [&,n = 0]() mutable { return dataframes[n].get_N_rows(); });
        return N_rows;
    }
}
