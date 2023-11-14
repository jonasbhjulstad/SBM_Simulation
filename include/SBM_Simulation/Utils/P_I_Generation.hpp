#ifndef SBM_SIMULATION_P_I_GENERATION_HPP
#define SBM_SIMULATION_P_I_GENERATION_HPP
#include <CL/sycl.hpp>
#include <QString>
#include <SBM_Database/Simulation/Sim_Types.hpp>
namespace SBM_Simulation {

sycl::buffer<float, 3>
generate_p_Is_excitation(sycl::queue &q, const SBM_Database::Sim_Param &p,
                         const QString &control_type);

sycl::buffer<float, 3>
generate_p_Is_validation(sycl::queue &q, const SBM_Database::Sim_Param &p,
                         const QString &control_type,
                         const QString &regression_type);

// std::vector<float> generate_floats(uint32_t N, float min, float max, uint32_t
// seed);

sycl::buffer<float, 3> generate_upsert_p_Is(sycl::queue &q,
                                            const SBM_Database::Sim_Param &p,
                                            const QString &table_name,
                                            const QString &control_type);
} // namespace SBM_Simulation

#endif