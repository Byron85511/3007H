#pragma once

#include <vector>

namespace lp_solver::model {

struct SolverState {
  std::vector<int> basic_indices;
  std::vector<int> nonbasic_indices;
  std::vector<double> x_basic;
  std::vector<double> reduced_costs;
  std::vector<double> dual_pi;
  /// Squared norms \(\|e_i^\top B^{-1}\|_2^2\) for dual steepest edge (see Goldfarb–Reid).
  std::vector<double> dse_weights;

  int iteration{0};
  double objective{0.0};
};

}  // namespace lp_solver::model
