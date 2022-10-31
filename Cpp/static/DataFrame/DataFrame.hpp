#ifndef FROLS_DATAFRAME_HPP
#define FROLS_DATAFRAME_HPP

#include <Typedefs.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace FROLS {
class DataFrameStack;

class DataFrame {
  std::map<std::string, std::shared_ptr<std::vector<float>>> data = {};
  uint32_t N_rows = 100;

  friend class DataFrameStack;

public:
  DataFrame() {}

  DataFrame(uint32_t N_rows) : N_rows(N_rows) {}

  DataFrame(const std::string &filename, const std::string delimiter = ",") {
    read_csv(filename, delimiter);
  }

  DataFrame(uint32_t N_rows, const std::vector<std::string> &col_names)
      : N_rows(N_rows) {
    for (auto &col_name : col_names) {
      data.insert({col_name, std::make_shared<std::vector<float>>(N_rows)});
    }
  }

  const std::shared_ptr<std::vector<float>>
  operator[](const std::string &col_name) const {
    assert(data.find(col_name) != data.end());
    return data.at(col_name);
  }

  const std::vector<std::vector<float>>
  operator[](const std::vector<std::string> &col_names) const;

  void assign(const std::string &col_name,
              const std::vector<uint32_t> &col_data);

  void assign(const std::string &col_name, const std::vector<float> &col_data);

  template <std::size_t N, typename T>
  void assign(const std::string &col_name, const std::array<T, N> &col_data) {

    data[col_name] = std::make_shared<std::vector<float>>(N);
    for (int i = 0; i < N; i++) {
      (*data[col_name])[i] = col_data[i];
    }
  }

  template <std::size_t N, std::size_t N_C, typename T>
  void assign(const std::vector<std::string> &col_names,
              const std::array<std::array<T, N>, N_C> &data_cols) {
    for (int i = 0; i < N_C; i++) {
      assign(col_names[i], data_cols[i]);
    }
  }

  void assign(const std::string &col_name, uint32_t row, float value);

  void assign(const std::string &col_name, Vec vec);

  // void assign(const std::vector<std::string> &col_names, crMat &mat);

  void assign(const std::vector<std::string> &col_names,
              const std::vector<std::vector<float>> &col_data);

  template <typename T>
  void assign(const std::vector<std::string> &col_names,
              const std::vector<std::vector<T>> &col_data) {
    for (uint32_t i = 0; i < col_names.size(); i++) {
      std::vector<float> col(col_data[i].size());
      std::copy(col_data[i].begin(), col_data[i].end(), col.begin());
      data[col_names[i]] = std::make_shared<std::vector<float>>(col);
    }
  }

  void assign(const std::vector<std::string> &col_names, Mat &mat);
  template <typename T> void assign(const char *col_name, const T &col_data) {
    assign(std::string(col_name), col_data);
  }

  template <typename T>
  void assign(const char **col_names, const std::vector<T> &col_data) {
    std::vector<std::string> col_names_str;
    for (int i = 0; i < col_data.size(); i++) {
      col_names_str.push_back(std::string(col_names[i]));
    }
    assign(col_names_str, col_data);
  }

  std::vector<std::string> get_col_names();

  void read_csv(const std::string &filename, const std::string &delimiter);

  void write_csv(
      const std::string &filename, const std::string &delimiter,
      const float termination_tol = -std::numeric_limits<float>::infinity());

  uint32_t get_N_rows() const { return N_rows; }

  void resize(uint32_t N_rows);

  void resize();
};

class DataFrameStack {

  friend class DataFrame;

public:
  std::vector<DataFrame> dataframes;
  DataFrameStack(const std::vector<std::string> &filenames) {
    dataframes.reserve(filenames.size());
    for (uint32_t i = 0; i < filenames.size(); i++) {
      dataframes.push_back(DataFrame(filenames[i]));
    }
  }

  DataFrameStack(const std::vector<DataFrame> &dataframes) {
    this->dataframes = dataframes;
  }

  DataFrameStack(uint32_t N_frames) { dataframes.resize(N_frames); }

  float get_elem(uint32_t frame, const std::string &col_name, uint32_t row) {
    return dataframes[frame][col_name]->operator[](row);
  }

  DataFrame &operator[](uint32_t frame) { return dataframes[frame]; }

  DataFrame compute_svd();

  const std::vector<float> get_col(uint32_t frame, const std::string &col_name);

  std::vector<float> get_row(uint32_t frame, uint32_t row);

  uint32_t get_N_frames() const { return dataframes.size(); }

  std::vector<uint32_t> get_N_rows();
};

} // namespace FROLS

#endif