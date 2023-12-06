#include <SBM_Simulation/Simulation/Sim_Infection_Sampling.hpp>
#include <tom/tom_config.hpp>
int main()
{
    using namespace SBM_Simulation;
    auto seed = 123;
    
    auto DB = tom_config::default_db_connection_postgres();
    Orm::DB::unprepared("DELETE FROM infection_events_excitation");
    sample_all_infections("Community", "", seed);
    return 0;
}