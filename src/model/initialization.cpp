#include "lp_solver/model/initialization.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace lp_solver::model {

namespace {

double dotColumn(const ProblemData& prob, const std::vector<double>& pi, int col) {
  const auto c = prob.A.column(col);
  double s = 0.0;
  for (int k = 0; k < c.numNonZeros(); ++k) {
    const int r = c.nonZeroIndices()[static_cast<size_t>(k)];
    s += pi[static_cast<size_t>(r)] * c.nonZeroValues()[static_cast<size_t>(k)];
  }
  return s;
}

}  // namespace

void initializeSlackBasis(const ProblemData& prob, SolverState& state) {
  const int m = prob.numRows();
  const int n = prob.numCols();
  if (n < m) {
    throw std::invalid_argument("initializeSlackBasis: need n >= m");
  }
  state.basic_indices.resize(static_cast<size_t>(m));
  state.nonbasic_indices.clear();
  for (int j = 0; j < n - m; ++j) {
    state.nonbasic_indices.push_back(j);
  }
  for (int i = 0; i < m; ++i) {
    const int col = n - m + i;
    state.basic_indices[static_cast<size_t>(i)] = col;
    const auto ac = prob.A.column(col);
    bool ok = false;
    for (int k = 0; k < ac.numNonZeros(); ++k) {
      const int r = ac.nonZeroIndices()[static_cast<size_t>(k)];
      const double v = ac.nonZeroValues()[static_cast<size_t>(k)];
      if (r == i && std::abs(v - 1.0) < 1e-10) {
        ok = true;
      }
      if (r != i && std::abs(v) > 1e-10) {
        throw std::invalid_argument("initializeSlackBasis: trailing block must be identity");
      }
    }
    if (!ok) {
      throw std::invalid_argument("initializeSlackBasis: trailing block must be identity");
    }
  }

  state.x_basic = prob.b;
  state.dual_pi.assign(static_cast<size_t>(m), 0.0);
  for (int i = 0; i < m; ++i) {
    state.dual_pi[static_cast<size_t>(i)] = prob.c[state.basic_indices[static_cast<size_t>(i)]];
  }
  state.reduced_costs.assign(static_cast<size_t>(n), 0.0);
  computeReducedCosts(prob, state.dual_pi, state);
  state.objective = computeObjective(prob, state);
}

void computeReducedCosts(const ProblemData& prob, const std::vector<double>& pi,
                         SolverState& state) {
  const int n = prob.numCols();
  state.reduced_costs.assign(static_cast<size_t>(n), 0.0);
  for (int j = 0; j < n; ++j) {
    state.reduced_costs[static_cast<size_t>(j)] =
        prob.c[static_cast<size_t>(j)] - dotColumn(prob, pi, j);
  }
}

double computeObjective(const ProblemData& prob, const SolverState& state) {
  double z = 0.0;
  const int m = prob.numRows();
  for (int i = 0; i < m; ++i) {
    z += prob.c[state.basic_indices[static_cast<size_t>(i)]] *
         state.x_basic[static_cast<size_t>(i)];
  }
  return z;
}

}  // namespace lp_solver::model
