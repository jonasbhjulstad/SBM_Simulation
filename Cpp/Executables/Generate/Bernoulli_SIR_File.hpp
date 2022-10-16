#ifndef BERNOULLI_SIR_FILE_HPP
#define BERNOULLI_SIR_FILE_HPP
#ifndef FROLS_USE_INTEL_SYCL
namespace FROLS
{
    template <size_t Nt, size_t NV, size_t NE, typename dType = float>
    void MC_SIR_to_file(const MC_SIR_Params<> &p, size_t thread_id)
    {

        static std::thread::id thread_0 = std::this_thread::get_id();

        FROLS::DataFrame df;
        FROLS::DataFrame delta_df;
        std::vector<std::string> colnames = {"S", "I", "R"};

        std::mt19937 generator(p.seed);

        thread_local Network_Models::SIR_Bernoulli_Network<decltype(generator), Nt, NV, NE> G(p.N_pop, p.p_ER, p.p_I0,
                                                                                              p.p_R0, generator);

        for (size_t i = 0; i < p.N_sim; i++)
        {
            if ((std::this_thread::get_id() == thread_0) && !(i % (p.N_sim / 10)))
            {
                std::cout << "Simulation " << i << " of " << p.N_sim << std::endl;
            }
            G.reset();
            while (G.population_count()[1] == 0)
            {
                G.initialize();
            }
            auto p_vec = generate_interaction_probabilities<decltype(generator), Nt>(p, generator);
            auto traj = G.simulate(p_vec);
            std::array<dType, Nt> p_Is;
            std::array<dType, Nt> p_Rs;
            std::fill(p_Rs.begin(), p_Rs.end(), p.p_R);
            std::transform(p_vec.begin(), p_vec.end(), p_Is.begin(), [](const auto &pv)
                           { return pv.p_I; });
            df.assign("S", traj[0]);
            df.assign("I", traj[1]);
            df.assign("R", traj[2]);
            df.assign("p_I", p_Is);
            df.assign("p_R", p_Rs);
            df.assign("t", FROLS::range(0, Nt + 1));
            df.write_csv(MC_filename(p.N_pop, p.p_ER, i + p.iter_offset, "SIR"),
                         ",", p.csv_termination_tol);

            delta_df.assign("S", diff<Nt + 1, size_t, int>(traj[0]));
            delta_df.assign("I", diff<Nt + 1, size_t, int>(traj[1]));
            delta_df.assign("R", diff<Nt + 1, size_t, int>(traj[2]));
            delta_df.assign("p_I", p_Is);
            delta_df.assign("p_R", p_Rs);
            delta_df.assign("t", FROLS::range(0, Nt));
            delta_df.write_csv(MC_filename(p.N_pop, p.p_ER, i + p.iter_offset, "SIR_Delta"),
                               ",", p.csv_termination_tol);
        }
    }

    template <size_t Nt, size_t NV, size_t NE, typename dType = float>
    Mat MC_SIR_to_Mat(const MC_SIR_Params<> &p)
    {

        static std::thread::id thread_0 = std::this_thread::get_id();
        thread_local std::mt19937 generator(p.seed);
        std::vector<Mat> Xi_vec;
        thread_local Network_Models::SIR_Bernoulli_Network<decltype(generator), Nt, NV, NE> G(p.N_pop, p.p_ER, p.p_I0,
                                                                                              p.p_R0, generator);
        for (size_t i = 0; i < p.N_sim; i++)
        {
            if ((std::this_thread::get_id() == thread_0) && !(i % (p.N_sim / 10)))
            {
                std::cout << "Simulation " << i << " of " << p.N_sim << std::endl;
            }
            G.reset();
            while (G.population_count()[1] == 0)
            {
                G.initialize(p.p_I0, p.p_R0);
            }
            auto p_vec = generate_interaction_probabilities<decltype(generator), Nt>(p, generator);
            auto traj = G.simulate(p_vec, p.N_I_min, p.Nt_min);
            auto [p_I_vec, p_R_vec] = FROLS::unzip(p_vec);
            std::array<dType, Nt> p_Is;
            std::transform(p_vec.begin(), p_vec.end(), p_Is.begin(), [](const auto &pv)
                           { return pv.p_I; });
            Mat Xi(Nt, 4);
            Xi.leftCols(3) = vecs_to_mat(traj);
            Xi.col(4) = Eigen::Map<Vec>(p_I_vec.data(), p_I_vec.size());
            Xi_vec.emplace_back(Xi);
        }
        size_t N_rows = 0;
        std::for_each(Xi_vec.begin(), Xi_vec.end(), [&](auto &xi)
                      { N_rows += xi.rows(); });
        Mat X(N_rows, 4);
        std::for_each(Xi_vec.begin(), Xi_vec.end(), [&](auto &xi)
                      { X << xi; });
        return X;
    }
}
#endif
#endif // FROLS_BERNOULLI_SIR_FILE_HPP