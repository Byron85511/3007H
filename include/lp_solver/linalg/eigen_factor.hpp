#pragma once

#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <vector>

#include "lp_solver/linalg/i_basis_factor.hpp"

namespace lp_solver::linalg {

/// Revised simplex basis operations using Eigen::SparseLU on \(B\) and an ETA file
/// for rank-one updates \(B' = B E\) with \(E = I + (d - e_p)e_p^\top\).
class EigenFactor : public IBasisFactor {
 public:
  [[nodiscard]] bool factorize(const util::PackedMatrix& basis_matrix) override;
  void ftran(util::IndexedVector& rhs) const override;
  void btran(util::IndexedVector& rhs) const override;
  void updateEta(int pivot_row, const util::IndexedVector& ftran_col) override;
  [[nodiscard]] int etaFileLength() const override;

  /// Drop ETA entries; next factorize() will rebuild LU only.
  void clearEtas() override;

 private:
  struct EtaEntry {
    int pivot_row{};
    std::vector<double> col;  // full length m, equals eta vector \(d\) from FTRAN
  };

  static void indexedToDense(const util::IndexedVector& v, int dim, Eigen::VectorXd& out);
  static void denseToIndexed(const Eigen::VectorXd& v, double zero_tol, util::IndexedVector& out);

  static void solveEtaEquation(int dim, int p, const std::vector<double>& d, util::IndexedVector& v);
  static void solveEtaTransposeEquation(int dim, int p, const std::vector<double>& d,
                                        util::IndexedVector& v);

  void solveLuOnly(util::IndexedVector& rhs) const;

  int dim_{0};
  bool factorized_{false};
  Eigen::SparseMatrix<double> basis_;
  mutable Eigen::SparseLU<Eigen::SparseMatrix<double>> lu_;
  std::vector<EtaEntry> etas_;
};

}  // namespace lp_solver::linalg
