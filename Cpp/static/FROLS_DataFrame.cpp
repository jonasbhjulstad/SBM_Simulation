#include "FROLS_DataFrame.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
namespace FROLS {


void DataFrame::assign(const std::string &col_name,
                       const std::vector<size_t> &col_data) {

  data[col_name] = std::make_shared<std::vector<double>>(col_data.size());
  for (int i = 0;i < col_data.size();i++) {
    (*data[col_name])[i] = col_data[i];
  }
    
  }

void DataFrame::assign(const std::string &col_name,
                       const std::vector<double> &col_data) {
  data[col_name] = std::make_shared<std::vector<double>>(col_data);
}

void DataFrame::assign(const std::string &col_name, size_t row, double value) {
  data[col_name]->operator[](row) = value;
}

void DataFrame::assign(const std::vector<std::string> &col_names,
            const std::vector<std::vector<size_t>> &col_data) {
  for (size_t i = 0; i < col_names.size(); i++) {
    std::vector<double> col(col_data[i].size());
    std::copy(col_data[i].begin(), col_data[i].end(), col.begin());
    data[col_names[i]] = std::make_shared<std::vector<double>>(col);
  }
}

void DataFrame::assign(const std::vector<std::string> &col_names,
            const std::vector<std::vector<double>> &col_data) {
  for (size_t i = 0; i < col_names.size(); i++) {
    *data[col_names[i]] = col_data[i];
  }
}

std::vector<std::string> DataFrame::get_col_names() {
  std::vector<std::string> col_names;
  for (auto &col : data) {
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
  for (auto &col_name : col_names) {
    std::shared_ptr<std::vector<double>> ptr =
        std::make_shared<std::vector<double>>(N_rows);
    data.insert({col_name, ptr});
  }
  size_t row = 0;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string value;
    size_t col = 0;
    while (std::getline(ss, value, delimiter[0])) {
      this->assign(col_names[col], row, std::stod(value));
      col++;
      if (col > N_rows) {
        N_rows += 100;
        for (auto &col_name : col_names) {
          this->data[col_name]->resize(N_rows);
        }
      }
    }
    row++;
  }
}
void DataFrame::write_csv(const std::string &filename,
               const std::string &delimiter) {
  std::ofstream file(filename);
  size_t N_cols = data.size();
  size_t iter = 0;
  for (auto &col : data) {
    iter++;
    file << col.first;
    if (iter < N_cols) {
      file << delimiter;
    }
  }
  file << "\n";
  for (size_t i = 0; i < N_rows; i++) {
    iter = 0;
    for (auto &col : data) {
      iter++;
      file << col.second->operator[](i);
      if (iter < N_cols) {
        file << delimiter;
      }
    }
    file << "\n";
  }
}

void DataFrame::resize(size_t N_rows) {
  if (!data.empty()) {
    for (auto &col : data) {
      col.second->resize(N_rows);
    }
  }
  this->N_rows = N_rows;
}


  const std::vector<double> DataFrameStack::get_col(size_t frame, const std::string &col_name) {
    assert(frame < dataframes.size());
    return *(dataframes[frame][col_name]);
  }

  std::vector<double> DataFrameStack::get_row(size_t frame, size_t row) {
    std::vector<double> row_data;
    for (auto &col : dataframes[frame].data) {
      row_data.push_back(col.second->operator[](row));
    }
    return row_data;
  }
}
