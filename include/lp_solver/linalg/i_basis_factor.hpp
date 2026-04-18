#pragma once

#include <memory>

#include "lp_solver/util/indexed_vector.hpp"
#include "lp_solver/util/packed_matrix.hpp"

namespace lp_solver::linalg {

class IBasisFactor {
 public:
  virtual ~IBasisFactor() = default;

  [[nodiscard]] virtual bool factorize(const util::PackedMatrix& basis_matrix) = 0;
  virtual void ftran(util::IndexedVector& rhs) const = 0;
  virtual void btran(util::IndexedVector& rhs) const = 0;

  virtual void updateEta(int pivot_row, const util::IndexedVector& ftran_col) {
    (void)pivot_row;
    (void)ftran_col;
  }

  [[nodiscard]] virtual int etaFileLength() const { return 0; }

  virtual void clearEtas() {}
};

enum class FactorBackend { Eigen, Umfpack };
std::unique_ptr<IBasisFactor> makeFactor(FactorBackend backend);

}  // namespace lp_solver::linalg
