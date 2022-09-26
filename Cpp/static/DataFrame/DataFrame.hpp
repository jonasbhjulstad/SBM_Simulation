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
        std::map<std::string, std::shared_ptr<std::vector<double>>> data = {};
        size_t N_rows = 100;

        friend class DataFrameStack;

    public:
        DataFrame() {}

        DataFrame(size_t N_rows) : N_rows(N_rows) {}

        DataFrame(const std::string &filename, const std::string delimiter = ",") { read_csv(filename, delimiter); }

        DataFrame(size_t N_rows, const std::vector<std::string> &col_names)
                : N_rows(N_rows) {
            for (auto &col_name: col_names) {
                data.insert({col_name, std::make_shared<std::vector<double>>(N_rows)});
            }
        }

        const std::shared_ptr<std::vector<double>>
        operator[](const std::string &col_name) const {
            assert(data.find(col_name) != data.end());
            return data.at(col_name);
        }

        void assign(const std::string &col_name, const std::vector<size_t> &col_data);

        void assign(const std::string &col_name, const std::vector<double> &col_data);

        template<size_t N, typename T>
        void assign(const std::string &col_name,
                    const std::array<T, N> &col_data) {

            data[col_name] = std::make_shared<std::vector<double>>(N);
            for (int i = 0; i < N; i++) {
                (*data[col_name])[i] = col_data[i];
            }
        }

        template<size_t N, size_t N_C, typename T>
        void assign(const std::vector<std::string> &col_names,
                    const std::array<std::array<T, N>, N_C> &data_cols) {
            for (int i = 0; i < N_C; i++) {
                assign(col_names[i], data_cols[i]);
            }
        }

        void assign(const std::string &col_name, size_t row, double value);

        void assign(const std::string &col_name, crVec &vec);

        void assign(const std::vector<std::string> &col_names, crMat &mat);

        void assign(const std::vector<std::string> &col_names,
                    const std::vector<std::vector<double>> &col_data);

        void assign(const std::vector<std::string> &col_names,
                    const std::vector<std::vector<size_t>> &col_data);


        std::vector<std::string> get_col_names();

        void read_csv(const std::string &filename,
                      const std::string &delimiter);

        void write_csv(const std::string &filename,
                       const std::string &delimiter,
                       const double termination_tol = -std::numeric_limits<double>::infinity());

        size_t get_N_rows() const { return N_rows; }

        void resize(size_t N_rows);

        void resize();
    };

    class DataFrameStack {

        friend class DataFrame;

    public:
        std::vector<DataFrame> dataframes;
        DataFrameStack(const std::vector<std::string> &filenames) {
            dataframes.reserve(filenames.size());
            for (size_t i = 0; i < filenames.size(); i++) {
                dataframes.push_back(DataFrame(filenames[i]));
            }
        }

        DataFrameStack(const std::vector<DataFrame> &dataframes) {
            this->dataframes = dataframes;
        }

        DataFrameStack(size_t N_frames)
        {
            dataframes.resize(N_frames);
        }

        double get_elem(size_t frame, const std::string &col_name, size_t row) {
            return dataframes[frame][col_name]->operator[](row);
        }



        DataFrame &operator[](size_t frame) { return dataframes[frame]; }

        DataFrame compute_svd();

        const std::vector<double> get_col(size_t frame, const std::string &col_name);

        std::vector<double> get_row(size_t frame, size_t row);

        size_t get_N_frames() const { return dataframes.size(); }

        std::vector<size_t> get_N_rows();
    };

} // namespace FROLS

#endif