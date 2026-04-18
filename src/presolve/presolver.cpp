#include "lp_solver/presolve/presolver.hpp"

#include <cmath>

namespace lp_solver::presolve {

namespace {

int rowNonZeroCount(const util::PackedMatrix& A, int row) {
  int cnt = 0;
  for (int j = 0; j < A.numCols(); ++j) {
    const auto col = A.column(j);
    for (int k = 0; k < col.numNonZeros(); ++k) {
      if (col.nonZeroIndices()[static_cast<size_t>(k)] == row) {
        ++cnt;
      }
    }
  }
  return cnt;
}

bool columnIsEmpty(const util::PackedMatrix& A, int col) { return A.column(col).numNonZeros() == 0; }

}  // namespace

PresolveResult presolveStandardForm(const model::ProblemData& prob) {
  PresolveResult out{prob, PresolveStatus::Ok};

  const int m = prob.numRows();
  for (int i = 0; i < m; ++i) {
    if (rowNonZeroCount(prob.A, i) == 0) {
      if (std::abs(prob.b[static_cast<size_t>(i)]) > 1e-12) {
        out.status = PresolveStatus::PrimalInfeasible;
        return out;
      }
    }
  }

  const int n = prob.numCols();
  for (int j = 0; j < n; ++j) {
    if (!columnIsEmpty(prob.A, j)) {
      continue;
    }
    const double cj = prob.c[static_cast<size_t>(j)];
    if (cj < -1e-12) {
      out.status = PresolveStatus::Unbounded;
      return out;
    }
  }

  return out;
}

void postsolvePrimal(const PresolveResult& meta, std::vector<double>& x_full) {
  (void)meta;
  (void)x_full;
}

}  // namespace lp_solver::presolve
