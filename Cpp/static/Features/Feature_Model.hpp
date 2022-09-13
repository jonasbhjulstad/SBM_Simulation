#ifndef FROLS_FEATURE_MODEL_HPP
#define FROLS_FEATURE_MODEL_HPP

#include <Typedefs.hpp>
#include <FROLS_Path_Config.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <memory>

namespace FROLS::Features {
    struct Feature_Model {

        Feature_Model();

        Vec step(crVec &x, crVec &u) const;

        Mat simulate(crVec &x0, crMat &U, size_t Nt) const;

        Vec transform(crMat &X_raw, size_t target_index) const;

        Mat transform(crMat &X_raw) const;

        virtual Vec _transform(crMat &X_raw, size_t target_index) const = 0;

        virtual Mat _transform(crMat &X_raw) const = 0;

        virtual const std::vector<std::vector<Feature>> get_features() const = 0;

        virtual void write_csv(const std::string &) const = 0;

        virtual void feature_summary() const = 0;

        virtual const std::string feature_name(size_t target_index,
                                               bool indent = true) const = 0;

        virtual const std::vector<std::string> feature_names() const = 0;

        virtual const std::string model_equation(size_t idx) const = 0;

        virtual const std::string model_equations() const = 0;

        std::vector<std::vector<Feature>> features;
    private:
        std::shared_ptr<spdlog::logger> feature_logger;
    };
} // namespace FROLS::Features

#endif // FEATURE_MODEL_HPP