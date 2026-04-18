#pragma once

#include "lp_solver/model/solver_state.hpp"
#include "lp_solver/simplex/i_row_pivot.hpp"

namespace lp_solver::simplex {

/// Dual steepest edge row pricing: maximize \(x_{B(i)}^2 / w_i\) over \(x_{B(i)}<0\).
class DseRowPivot final : public IRowPivot {
 public:
  explicit DseRowPivot(double primal_tol = 1e-7) : tol_(primal_tol) {}

  [[nodiscard]] int chooseRow(const model::SolverState& state) const override {
    const int m = static_cast<int>(state.x_basic.size());
    if (static_cast<int>(state.dse_weights.size()) != m) {
      return -1;
    }
    int best = -1;
    double best_score = -1.0;
    for (int i = 0; i < m; ++i) {
      const double xb = state.x_basic[static_cast<size_t>(i)];
      if (xb >= -tol_) {
        continue;
      }
      const double w = state.dse_weights[static_cast<size_t>(i)];
      if (w <= 1e-30) {
        continue;
      }
      const double score = (xb * xb) / w;
      if (best < 0 || score > best_score) {
        best = i;
        best_score = score;
      }
    }
    return best;
  }

 private:
  double tol_;
};

}  // namespace lp_solver::simplex
