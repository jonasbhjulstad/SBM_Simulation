#include <casadi/casadi.hpp>
#include <Sycl_Graph/Utils/json_settings.hpp>
#include <fstream>
using namespace casadi;

auto read_pairs(const std::string &fname)
{
    std::ifstream file(fname);
    // for each line
    std::string line;
    std::vector<std::pair<uint32_t, uint32_t>> ccm;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string token;
        std::vector<uint32_t> tokens;
        while (std::getline(ss, token, ','))
        {
            tokens.push_back(std::stoi(token));
        }
        ccm.push_back({tokens[0], tokens[1]});
    }
    return ccm;
}

std::vector<float> read_float_line(const std::string& fname)
{
    std::ifstream file(fname);
    // for each line
    std::string line;
    std::vector<std::pair<uint32_t, uint32_t>> ccm;
    auto line = std::getline(file, line);
    std::stringstream ss(line);
    std::string token;
    std::vector<float> tokens;
    while (std::getline(ss, token, ','))
    {
        tokens.push_back(std::stof(token));
    }
    return tokens;
}

struct Regression_Parameter_Set_t
{
    std::vector<float> theta_LS;
    std::vector<float> theta_QR;

};

auto diff(auto& x)
{
    return x(Slice(1, None)) - x(Slice(None, -1));
}

auto delta_I(auto& state_s, auto&state_r, auto& p_I, auto theta)
{
    return p_I*state_s[0]*state_r[1]/(state_s[0] + state_r[1] + state_s[2])*theta;
}

auto c_delta_I(auto c_idx, auto c_state, auto c_p_I, const auto& ccm, const auto& betas)
{
    auto N_connections = ccm.size();
    MX d_I = 0;
    auto state_s = c_state(Slice(3*c_idx, 3*c_idx+3));
    auto state_r = state_s;
    auto idx = 0;
    for (const auto&& cm : ccm)
    {
        if (cm.first == c_idx)
        {
            auto beta_c = betas[2*idx];
            state_r = c_state(Slice(3*cm.second, 3*cm.second+3));
            d_I += delta_I(state_s, state_r, c_p_I, beta_c);
        }
        idx++;
    }
    return d_I;
}

auto c_delta_R(auto& c_state)
{
    //every third element
    return c_state(Slice(1, None, 3));
}

auto construct_objective(const Sim_Param& p, auto& F, auto& u, auto x0, auto Nt_per_u, auto u_max)
{
    MX f = 0;
    std::vector<MX> x_vec(Nt + 1);
    x_vec[0] = DM(x0);
    MX ut;
    for(int t = 0; t < p.Nt; t++)
    {
        ut = u(Slice(floor_div(t, Nt_per_u)), Slice()));
        x_vec[t] =  x_vec[t-1] + F(x_vec[t-1], ut);
        auto inf_sum = 0;
        for(int i = 0; i < p.N_communities; i++)
        {
            inf_sum += x_vec[t][3*i];
        }
        f += inf_sum;
        for (int k = 0; k < p.N_connections; k++)
        {
            f += Wu/p.N_pop*(ut[k] - u_max);
        }

    }
    return std::make_tuple(f, x_vec);
}

int main()
{

    auto ccm = read_pairs(Sycl_Graph::SYCL_GRAPH_DATA_DIR + "/Graph_0/ccm.csv");
    auto p = parse_json(Sycl_Graph::SYCL_GRAPH_DATA_DIR + "/Sim_Param.json");
    p.N_connections = ccm.size();

    MX c_state = MX::sym("Community_State", 3*p.N_communities);

    MX p_I = MX::sym("p_I", p.N_connections);

    std::vector<MX> c_delta_Is_vec(p.N_communities);
    std::generate_n(c_delta_Is.begin(), p.N_communities, [&, n = 0]()mutable {return c_delta_I(n, c_state, p_I);});
    auto c_delta_Rs = c_delta_R(c_state);
    auto c_delta_Is = MX::vertcat(c_delta_Is_vec);

    //merge the deltas together
    MX delta("dx", 3*p.N_communities);
    for(int i = 0; i < p.N_communities; i++)
    {
        delta(Slice(3*i, 3*i+3)) = MX::vertcat({c_delta_Is[i], c_delta_Is[i] - c_delta_Rs[i], c_delta_Rs[i]});
    }

    auto f = Function("f", {c_state, p_I}, {delta});

    const auto Nt_per_u = 7;
    auto Nu = floor_div(p.Nt, Nt_per_u);
    auto u = MX::sym("u", Nu, p.N_connections);

    auto [f, state] = construct_objective(p, F, u, x0, Nt_per_u, u_max);

    auto w = reshape(u, Nu*p.N_connections, 1);
    auto f_w_u = Function("f_w_u", {w}, {u});
    auto f_traj = Function("f_traj", {w}, horzcat(state));


    return 0;
}
