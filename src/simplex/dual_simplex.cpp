#include "lp_solver/simplex/dual_simplex.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "lp_solver/model/initialization.hpp"

namespace lp_solver::simplex {

namespace {

double dotColumnWithPi(const model::ProblemData& prob, const util::IndexedVector& pi_vec, int col) {
  const auto c = prob.A.column(col);
  double s = 0.0;
  for (int k = 0; k < c.numNonZeros(); ++k) {
    const int r = c.nonZeroIndices()[static_cast<size_t>(k)];
    s += pi_vec[r] * c.nonZeroValues()[static_cast<size_t>(k)];
  }
  return s;
}

void syncPiAndReducedCosts(const model::ProblemData& prob, linalg::IBasisFactor& factor,
                           model::SolverState& state) {
  const int m = prob.numRows();
  util::IndexedVector cb(m);
  for (int i = 0; i < m; ++i) {
    cb.set(i, prob.c[state.basic_indices[static_cast<size_t>(i)]]);
  }
  factor.btran(cb);
  state.dual_pi.resize(static_cast<size_t>(m));
  for (int i = 0; i < m; ++i) {
    state.dual_pi[static_cast<size_t>(i)] = cb[i];
  }
  model::computeReducedCosts(prob, state.dual_pi, state);
  state.objective = model::computeObjective(prob, state);
}

void goldfarbReidUpdate(int pivot_row, const util::IndexedVector& d, const util::IndexedVector& v,
                        model::SolverState& state) {
  const int m = static_cast<int>(state.dse_weights.size());
  const double dp = d[pivot_row];
  if (std::abs(dp) < std::numeric_limits<double>::epsilon() * 1e8) {
    return;
  }
  const double inv = 1.0 / dp;
  const double wp = state.dse_weights[static_cast<size_t>(pivot_row)];
  state.dse_weights[static_cast<size_t>(pivot_row)] = std::max(1.0, wp * inv * inv);
  for (int k = 0; k < d.numNonZeros(); ++k) {
    const int i = d.nonZeroIndices()[static_cast<size_t>(k)];
    if (i == pivot_row) {
      continue;
    }
    const double di = d[i];
    const double ratio = di * inv;
    const double vi = v[i];
    const double wi = state.dse_weights[static_cast<size_t>(i)];
    const double new_w = wi - 2.0 * ratio * vi + ratio * ratio * wp;
    state.dse_weights[static_cast<size_t>(i)] = std::max(1.0, new_w);
  }
}

}  // namespace

DualSimplex::DualSimplex(std::unique_ptr<linalg::IBasisFactor> factor,
                         std::unique_ptr<IRowPivot> row_pivot, ISolverObserver* observer)
    : factor_(std::move(factor)), row_pivot_(std::move(row_pivot)), observer_(observer) {}

util::PackedMatrix DualSimplex::buildBasisMatrix(const model::ProblemData& prob,
                                                 const model::SolverState& state) {
  const int m = prob.numRows();
  util::PackedMatrix::Builder builder(m, m);
  for (int k = 0; k < m; ++k) {
    const int col = state.basic_indices[static_cast<size_t>(k)];
    const auto ac = prob.A.column(col);
    std::vector<int> rows;
    std::vector<double> vals;
    rows.reserve(static_cast<size_t>(ac.numNonZeros()));
    vals.reserve(static_cast<size_t>(ac.numNonZeros()));
    for (int t = 0; t < ac.numNonZeros(); ++t) {
      rows.push_back(ac.nonZeroIndices()[static_cast<size_t>(t)]);
      vals.push_back(ac.nonZeroValues()[static_cast<size_t>(t)]);
    }
    builder.appendColumn(rows, vals);
  }
  return builder.build();
}

void DualSimplex::refactorBasis(const model::ProblemData& prob, model::SolverState& state) {
  const auto basis = buildBasisMatrix(prob, state);
  if (!factor_->factorize(basis)) {
    throw std::runtime_error("refactorBasis: factorize failed");
  }
  syncPiAndReducedCosts(prob, *factor_, state);
}

void DualSimplex::initializeDseWeights(const model::ProblemData& prob,
                                      model::SolverState& state) const {
  const int m = prob.numRows();
  state.dse_weights.assign(static_cast<size_t>(m), 1.0);
  for (int i = 0; i < m; ++i) {
    util::IndexedVector ei(m);
    ei.set(i, 1.0);
    factor_->btran(ei);
    double w = 0.0;
    for (int r = 0; r < m; ++r) {
      const double t = ei[r];
      w += t * t;
    }
    state.dse_weights[static_cast<size_t>(i)] = std::max(1.0, w);
  }
}

DualSimplex::Status DualSimplex::solve(const model::ProblemData& prob, model::SolverState& state,
                                       const SolverConfig& cfg) {
  const int m = prob.numRows();
  const int n = prob.numCols();
  if (m <= 0 || n <= 0) {
    return Status::Infeasible;
  }
  if (static_cast<int>(state.basic_indices.size()) != m ||
      static_cast<int>(state.x_basic.size()) != m) {
    throw std::invalid_argument("DualSimplex::solve: invalid basis state dimensions");
  }
  if (static_cast<int>(prob.c.size()) != n) {
    throw std::invalid_argument("DualSimplex::solve: objective length mismatch");
  }

  refactorBasis(prob, state);
  if (cfg.use_dual_steepest_edge) {
    initializeDseWeights(prob, state);
  } else {
    state.dse_weights.clear();
  }

  for (int iter = 0; iter < cfg.max_iterations; ++iter) {
    state.iteration = iter;
    if (observer_ != nullptr) {
      observer_->onIterationBegin(state);
    }
    const int leaving_row = chuzr(state);
    if (leaving_row < 0) {
      const bool dual_feasible =
          std::all_of(state.reduced_costs.begin(), state.reduced_costs.end(),
                      [&](double rc) { return rc >= -cfg.dual_feasibility_tol; });
      if (observer_ != nullptr) {
        observer_->onIterationEnd(state);
        observer_->onTermination(state, dual_feasible ? "optimal" : "dual_infeasible");
      }
      return dual_feasible ? Status::Optimal : Status::DualInfeasible;
    }

    util::IndexedVector ep(m);
    ep.set(leaving_row, 1.0);
    btran(ep);

    util::IndexedVector btran_ep(m);
    for (int i = 0; i < m; ++i) {
      btran_ep.set(i, ep[i]);
    }

    const int entering_col = chuzc(ep, prob, state, cfg);
    if (entering_col < 0) {
      if (observer_ != nullptr) {
        observer_->onIterationEnd(state);
        observer_->onTermination(state, "primal_infeasible");
      }
      return Status::Infeasible;
    }

    util::IndexedVector aq = prob.A.column(entering_col);
    ftran(aq);

    const double d_p = aq[leaving_row];
    if (d_p >= -cfg.primal_feasibility_tol) {
      if (observer_ != nullptr) {
        observer_->onIterationEnd(state);
        observer_->onTermination(state, "numerical_breakdown");
      }
      return Status::Infeasible;
    }

    const double t = state.x_basic[static_cast<size_t>(leaving_row)] / d_p;
    for (int i = 0; i < m; ++i) {
      const double di = aq[i];
      if (di != 0.0) {
        state.x_basic[static_cast<size_t>(i)] -= t * di;
      }
    }

    util::IndexedVector v_for_gr(m);
    if (cfg.use_dual_steepest_edge) {
      for (int i = 0; i < m; ++i) {
        v_for_gr.set(i, btran_ep[i]);
      }
      ftran(v_for_gr);
    }

    factor_->updateEta(leaving_row, aq);
    pivot(leaving_row, entering_col, aq, btran_ep, prob, state, cfg);

    if (factor_->etaFileLength() >= cfg.refactor_frequency) {
      refactorBasis(prob, state);
      if (cfg.use_dual_steepest_edge) {
        initializeDseWeights(prob, state);
      }
    } else {
      syncPiAndReducedCosts(prob, *factor_, state);
      if (cfg.use_dual_steepest_edge) {
        goldfarbReidUpdate(leaving_row, aq, v_for_gr, state);
      }
    }

    if (observer_ != nullptr) {
      observer_->onIterationEnd(state);
    }
  }

  if (observer_ != nullptr) {
    observer_->onTermination(state, "iteration_limit");
  }
  return Status::IterationLimit;
}

int DualSimplex::chuzr(const model::SolverState& state) const {
  return row_pivot_->chooseRow(state);
}

void DualSimplex::btran(util::IndexedVector& ep) const { factor_->btran(ep); }

int DualSimplex::chuzc(const util::IndexedVector& pivot_row, const model::ProblemData& prob,
                        const model::SolverState& state, const SolverConfig& cfg) const {
  const int n = prob.numCols();
  std::vector<char> is_basic(static_cast<size_t>(n), 0);
  for (int bi : state.basic_indices) {
    is_basic[static_cast<size_t>(bi)] = 1;
  }

  const double eps = cfg.dual_feasibility_tol;
  int best_col = -1;
  double best_ratio = std::numeric_limits<double>::infinity();

  auto scan_alpha = [&](double& out_delta) {
    out_delta = std::numeric_limits<double>::infinity();
    bool any = false;
    for (int j = 0; j < n; ++j) {
      if (is_basic[static_cast<size_t>(j)] != 0) {
        continue;
      }
      const double alpha = dotColumnWithPi(prob, pivot_row, j);
      if (alpha >= -eps) {
        continue;
      }
      any = true;
      const double rc = state.reduced_costs[static_cast<size_t>(j)];
      const double ratio = (rc + eps) / std::abs(alpha);
      out_delta = std::min(out_delta, ratio);
    }
    return any;
  };

  double delta = 0.0;
  if (!scan_alpha(delta)) {
    return -1;
  }

  if (!cfg.use_harris_ratio_test) {
    bool first = true;
    for (int j = 0; j < n; ++j) {
      if (is_basic[static_cast<size_t>(j)] != 0) {
        continue;
      }
      const double alpha = dotColumnWithPi(prob, pivot_row, j);
      if (alpha >= -eps) {
        continue;
      }
      const double rc = state.reduced_costs[static_cast<size_t>(j)];
      const double ratio = rc / std::abs(alpha);
      if (first || ratio < best_ratio - 1e-15) {
        first = false;
        best_ratio = ratio;
        best_col = j;
      }
    }
    return best_col;
  }

  double best_pivot_abs = -1.0;
  for (int j = 0; j < n; ++j) {
    if (is_basic[static_cast<size_t>(j)] != 0) {
      continue;
    }
    const double alpha = dotColumnWithPi(prob, pivot_row, j);
    if (alpha >= -eps) {
      continue;
    }
    const double rc = state.reduced_costs[static_cast<size_t>(j)];
    const double harris_ratio = (rc + eps) / std::abs(alpha);
    if (harris_ratio > delta + 1e-9) {
      continue;
    }
    const double piv = std::abs(alpha);
    if (piv > best_pivot_abs) {
      best_pivot_abs = piv;
      best_col = j;
    }
  }
  return best_col;
}

void DualSimplex::ftran(util::IndexedVector& aq) const { factor_->ftran(aq); }

void DualSimplex::pivot(int leaving_row, int entering_col, const util::IndexedVector& ftran_col,
                        const util::IndexedVector& btran_row, const model::ProblemData& prob,
                        model::SolverState& state, const SolverConfig& cfg) {
  (void)ftran_col;
  (void)btran_row;
  (void)cfg;
  const int leaving_basic_col = state.basic_indices[static_cast<size_t>(leaving_row)];
  state.basic_indices[static_cast<size_t>(leaving_row)] = entering_col;

  auto& nb = state.nonbasic_indices;
  nb.erase(std::remove(nb.begin(), nb.end(), entering_col), nb.end());
  nb.push_back(leaving_basic_col);

  (void)prob;
}

}  // namespace lp_solver::simplex
