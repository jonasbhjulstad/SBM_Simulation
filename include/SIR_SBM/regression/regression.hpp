#pragma once

#include <casadi/casadi.hpp>
#include <filesystem>
#include <tuple>


namespace SIR_SBM {

std::tuple<casadi::DM, casadi::DM>
connection_expand_population(const std::tuple<casadi::DM, casadi::DM> &data,
                             uint32_t N_connections);
// load csv into MX matrix
std::tuple<casadi::DM, casadi::DM, casadi::DM>
regression_data_from_simulations(const std::filesystem::path &filenameprefix,
                                 uint32_t N_communities, uint32_t N_connections,
                                 uint32_t N_sims, uint32_t Nt);
} // namespace SIR_SBM