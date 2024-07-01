// regression.hpp
//

#ifndef LZZ_SIR_SBM_Regression_LZZ_regression_hpp
#define LZZ_SIR_SBM_Regression_LZZ_regression_hpp
#line 3 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM/regression//regression.hpp"
#include <SIR_SBM/csv.hpp>
#include <casadi/casadi.hpp>
#include <filesystem>
#include <fstream>
#include <tuple>
#define LZZ_INLINE inline
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM/regression//regression.hpp"
namespace SIR_SBM
{
#line 18 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM/regression//regression.hpp"
  std::tuple <casadi::DM, casadi::DM> connection_expand_population (std::tuple <casadi::DM, casadi::DM> const & data, size_t N_connections);
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM/regression//regression.hpp"
namespace SIR_SBM
{
#line 50 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM/regression//regression.hpp"
  std::tuple <casadi::DM, casadi::DM, casadi::DM> regression_data_from_simulations (std::filesystem::path const & filenameprefix, size_t N_communities, size_t N_connections, size_t N_sims, size_t Nt);
}
#undef LZZ_INLINE
#endif
