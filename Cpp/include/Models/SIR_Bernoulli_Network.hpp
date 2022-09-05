#ifndef SIR_BERNOULLI_NETWORK_HPP
#define SIR_BERNOULLI_NETWORK_HPP

#include <array>
#include <igraph/igraph.h>
#include <igraph/igraph_games.h>
#include <random>
#include <stddef.h>
#include <utility>
#include <vector>
enum SIR_State { S = 0, I = 1, R = 2 };
struct SIR_Trajectory
{
  void resize(size_t N)
  {
    S.resize(N);
    I.resize(N);
    R.resize(N);
    p_I.resize(N);
  }
  std::vector<double> S;
  std::vector<double> I;
  std::vector<double> R;
  std::vector<double> p_I;
  const std::vector<std::string> colnames = {"S", "I", "R"};
};


template <typename RNG>
std::vector<SIR_State> generate_SIR_ER_model(igraph_t &G, size_t N_pop,
                                             double p_ER, double p_I0,
                                             RNG &rng) {
  igraph_erdos_renyi_game_gnp(&G, N_pop, p_ER, false, false);

  std::vector<SIR_State> state(N_pop);
  std::bernoulli_distribution d(p_I0);
  for (size_t i = 0; i < N_pop; i++) {
    state[i] = d(rng) ? I : S;
  }

  return state;
}

inline std::array<size_t, 3> count_SIR_state(const std::vector<SIR_State> &state) {
  std::array<size_t, 3> count = {0, 0, 0};
  for (auto s : state) {
    count[s]++;
  }
  return count;
}

// function for infection step
template <typename RNG>
std::vector<SIR_State> infection_step(const igraph_t &G,
                                      const std::vector<SIR_State> &state,
                                      double p_infect, RNG &rng) {
  std::vector<SIR_State> new_state = state;
  std::bernoulli_distribution d(p_infect);
  for (size_t i = 0; i < state.size(); i++) {
    if (state[i] == I) {
      igraph_vector_t neighbors;
      igraph_vector_init(&neighbors, 0);
      igraph_neighbors(&G, &neighbors, i, IGRAPH_ALL);
      for (size_t j = 0; j < igraph_vector_size(&neighbors); j++) {
        size_t neighbor = (size_t)VECTOR(neighbors)[j];
        if (state[neighbor] == S && d(rng)) {
          new_state[neighbor] = I;
        }
      }
      igraph_vector_destroy(&neighbors);
    }
  }
  return new_state;
}

template <typename RNG>
std::vector<SIR_State> recovery_step(const std::vector<SIR_State> &state,
                                     double p_recover, RNG &rng) {
  std::vector<SIR_State> new_state = state;
  std::bernoulli_distribution d(p_recover);
  for (size_t i = 0; i < state.size(); i++) {
    if (state[i] == I && d(rng)) {
      new_state[i] = R;
    }
  }
  return new_state;
}

template <typename RNG>
std::vector<std::vector<size_t>>
run_SIR_simulation(igraph_t &G, const std::vector<SIR_State> x0, size_t Nt,
                   const std::vector<double> &p_I,
                   const std::vector<double> &p_R, RNG &rng) {

  std::vector<std::vector<size_t>> SIR_trajectory(3);
  SIR_trajectory[S].resize(Nt+1);
  SIR_trajectory[I].resize(Nt+1);
  SIR_trajectory[R].resize(Nt+1);

  std::vector<std::vector<SIR_State>> graph_states(Nt + 1);
  graph_states[0] = x0;

  auto state = count_SIR_state(x0);
  SIR_trajectory[0][0] = state[0];
  SIR_trajectory[1][0] = state[1];
  SIR_trajectory[2][0] = state[2];

  for (int i = 1; i < Nt + 1; i++) {
    graph_states[i] = infection_step(G, graph_states[i - 1], p_I[i - 1], rng);
    graph_states[i] = recovery_step(graph_states[i], p_R[i - 1], rng);
    auto state = count_SIR_state(graph_states[i]);
    SIR_trajectory[0][i] = state[0];
    SIR_trajectory[1][i] = state[1];
    SIR_trajectory[2][i] = state[2];
  }
  return SIR_trajectory;
}

#endif