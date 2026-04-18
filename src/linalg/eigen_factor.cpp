#include "lp_solver/linalg/eigen_factor.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lp_solver::linalg {

namespace {

constexpr double kZeroTol = 1e-14;

Eigen::SparseMatrix<double> packedToEigen(const util::PackedMatrix& B) {
  const int m = B.numRows();
  const int n = B.numCols();
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(static_cast<size_t>(std::max(1, B.numNonZeros())));
  for (int j = 0; j < n; ++j) {
    const util::IndexedVector col = B.column(j);
    const auto idx = col.nonZeroIndices();
    const auto vals = col.nonZeroValues();
    for (int k = 0; k < col.numNonZeros(); ++k) {
      triplets.emplace_back(idx[static_cast<size_t>(k)], j, vals[static_cast<size_t>(k)]);
    }
  }
  Eigen::SparseMatrix<double> mat(m, n);
  mat.setFromTriplets(triplets.begin(), triplets.end());
  mat.makeCompressed();
  return mat;
}

}  // namespace

bool EigenFactor::factorize(const util::PackedMatrix& basis_matrix) {
  etas_.clear();
  if (basis_matrix.numRows() != basis_matrix.numCols()) {
    factorized_ = false;
    return false;
  }
  dim_ = basis_matrix.numRows();
  basis_ = packedToEigen(basis_matrix);
  lu_.compute(basis_);
  if (lu_.info() != Eigen::Success) {
    factorized_ = false;
    return false;
  }
  factorized_ = true;
  return true;
}

void EigenFactor::indexedToDense(const util::IndexedVector& v, int dim, Eigen::VectorXd& out) {
  out.setZero(dim);
  for (int k = 0; k < v.numNonZeros(); ++k) {
    const int i = v.nonZeroIndices()[static_cast<size_t>(k)];
    out[i] = v[i];
  }
}

void EigenFactor::denseToIndexed(const Eigen::VectorXd& v, double zero_tol,
                                 util::IndexedVector& out) {
  out.clear();
  for (int i = 0; i < v.size(); ++i) {
    if (std::abs(v[i]) > zero_tol) {
      out.set(i, v[i]);
    }
  }
}

void EigenFactor::solveLuOnly(util::IndexedVector& rhs) const {
  Eigen::VectorXd b;
  indexedToDense(rhs, dim_, b);
  const Eigen::VectorXd x = lu_.solve(b);
  if (!x.allFinite()) {
    throw std::runtime_error("LU solve produced non-finite values");
  }
  denseToIndexed(x, kZeroTol, rhs);
}

void EigenFactor::solveEtaEquation(int dim, int p, const std::vector<double>& d,
                                   util::IndexedVector& v) {
  const double dp = d[static_cast<size_t>(p)];
  if (std::abs(dp) < std::numeric_limits<double>::epsilon() * 1e8) {
    throw std::runtime_error("eta pivot too small in FTRAN");
  }
  std::vector<double> rhs(static_cast<size_t>(dim));
  for (int i = 0; i < dim; ++i) {
    rhs[static_cast<size_t>(i)] = v[i];
  }
  const double xp = rhs[static_cast<size_t>(p)] / dp;
  for (int i = 0; i < dim; ++i) {
    if (i == p) {
      continue;
    }
    const double nv = rhs[static_cast<size_t>(i)] - d[static_cast<size_t>(i)] * xp;
    v.set(i, nv);
  }
  v.set(p, xp);
}

void EigenFactor::solveEtaTransposeEquation(int dim, int p, const std::vector<double>& d,
                                            util::IndexedVector& v) {
  double sum_d2 = 0.0;
  double sum_dv = 0.0;
  for (int j = 0; j < dim; ++j) {
    if (j == p) {
      continue;
    }
    const double dj = d[static_cast<size_t>(j)];
    sum_d2 += dj * dj;
    sum_dv += dj * v[j];
  }
  const double denom = d[static_cast<size_t>(p)] - sum_d2;
  if (std::abs(denom) < std::numeric_limits<double>::epsilon() * 1e8) {
    throw std::runtime_error("eta transpose pivot too small in BTRAN");
  }
  std::vector<double> old(static_cast<size_t>(dim));
  for (int i = 0; i < dim; ++i) {
    old[static_cast<size_t>(i)] = v[i];
  }
  const double zp = (old[static_cast<size_t>(p)] - sum_dv) / denom;
  v.set(p, zp);
  for (int j = 0; j < dim; ++j) {
    if (j == p) {
      continue;
    }
    v.set(j, old[static_cast<size_t>(j)] - d[static_cast<size_t>(j)] * zp);
  }
}

void EigenFactor::ftran(util::IndexedVector& rhs) const {
  if (!factorized_) {
    throw std::runtime_error("factorize() must be called before ftran()");
  }
  solveLuOnly(rhs);
  for (const auto& e : etas_) {
    solveEtaEquation(dim_, e.pivot_row, e.col, rhs);
  }
}

void EigenFactor::btran(util::IndexedVector& rhs) const {
  if (!factorized_) {
    throw std::runtime_error("factorize() must be called before btran()");
  }
  for (int k = static_cast<int>(etas_.size()) - 1; k >= 0; --k) {
    const auto& e = etas_[static_cast<size_t>(k)];
    solveEtaTransposeEquation(dim_, e.pivot_row, e.col, rhs);
  }
  Eigen::VectorXd b;
  indexedToDense(rhs, dim_, b);
  const Eigen::VectorXd x = lu_.transpose().solve(b);
  if (!x.allFinite()) {
    throw std::runtime_error("LU transpose solve produced non-finite values");
  }
  denseToIndexed(x, kZeroTol, rhs);
}

void EigenFactor::updateEta(int pivot_row, const util::IndexedVector& ftran_col) {
  if (!factorized_) {
    throw std::runtime_error("factorize() before updateEta()");
  }
  if (pivot_row < 0 || pivot_row >= dim_) {
    throw std::out_of_range("pivot_row out of range for updateEta");
  }
  EtaEntry e;
  e.pivot_row = pivot_row;
  e.col.assign(static_cast<size_t>(dim_), 0.0);
  for (int k = 0; k < ftran_col.numNonZeros(); ++k) {
    const int i = ftran_col.nonZeroIndices()[static_cast<size_t>(k)];
    e.col[static_cast<size_t>(i)] = ftran_col[i];
  }
  etas_.push_back(std::move(e));
}

int EigenFactor::etaFileLength() const { return static_cast<int>(etas_.size()); }

void EigenFactor::clearEtas() { etas_.clear(); }

}  // namespace lp_solver::linalg
