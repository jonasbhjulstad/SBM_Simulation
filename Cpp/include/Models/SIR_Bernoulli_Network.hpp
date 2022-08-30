#ifndef SIR_BERNOULLI_NETWORK_HPP
#define SIR_BERNOULLI_NETWORK_HPP

#include <igraph/igraph.h>
#include <igraph/igraph_games.h>
#include <stddef.h>
#include <utility>
#include <array>
#include <vector>
#include <random>
enum SIR_State
{
  S = 0,
  I = 1,
  R = 2
};

template <typename RNG>
std::vector<SIR_State>
generate_SIR_ER_model(igraph_t &G, size_t N_pop, double p_ER, double p_I0, RNG &rng)
{
  igraph_erdos_renyi_game_gnp(&G, N_pop, p_ER, false, false);

  std::vector<SIR_State> state(N_pop);
  std::bernoulli_distribution d(p_I0);
  for (size_t i = 0; i < N_pop; i++)
  {
    state[i] = d(rng) ? I : S;
  }

  return state;
}

std::array<size_t, 3> count_SIR_state(const std::vector<SIR_State> &state)
{
  std::array<size_t, 3> count = {0, 0, 0};
  for (auto s : state)
  {
    count[s]++;
  }
  return count;
}

// function for infection step
template <typename RNG>
std::vector<SIR_State> infection_step(const igraph_t &G,
                                      const std::vector<SIR_State> &state,
                                      double p_infect,
                                      RNG &rng)
{
  std::vector<SIR_State> new_state = state;
  std::bernoulli_distribution d(p_infect);
  for (size_t i = 0; i < state.size(); i++)
  {
    if (state[i] == I)
    {
      igraph_vector_t neighbors;
      igraph_vector_init(&neighbors, 0);
      igraph_neighbors(&G, &neighbors, i, IGRAPH_ALL);
      for (size_t j = 0; j < igraph_vector_size(&neighbors); j++)
      {
        size_t neighbor = (size_t)VECTOR(neighbors)[j];
        if (state[neighbor] == S && d(rng))
        {
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
                                     double p_recover,
                                     RNG &rng)
{
  std::vector<SIR_State> new_state = state;
  std::bernoulli_distribution d(p_recover);
  for (size_t i = 0; i < state.size(); i++)
  {
    if (state[i] == I && d(rng))
    {
      new_state[i] = R;
    }
  }
  return new_state;
}

template <typename RNG>
std::vector<std::array<size_t, 3>>
run_SIR_simulation(igraph_t &G, const std::vector<SIR_State> x0, size_t Nt,
                   const std::vector<double> &p_I, const std::vector<double> &p_R,
                   RNG& rng)
{

  std::vector<std::array<size_t, 3>> SIR_counts(Nt + 1);
  std::vector<std::vector<SIR_State>> graph_states(Nt + 1);
  graph_states[0] = x0;
  SIR_counts[0] = count_SIR_state(x0);
  for (int i = 1; i < Nt + 1; i++)
  {
    graph_states[i] = infection_step(G, graph_states[i - 1], p_I[i - 1], rng);
    graph_states[i] = recovery_step(graph_states[i], p_R[i - 1], rng);
    SIR_counts[i] = count_SIR_state(graph_states[i]);
  }
  return SIR_counts;
}

#endif