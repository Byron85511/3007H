#pragma once

#include "lp_solver/linalg/eigen_factor.hpp"

namespace lp_solver::linalg {

/// Second backend hook: currently uses the same Eigen SparseLU implementation as
/// \ref EigenFactor so the project builds without an external SuiteSparse install.
class UmfpackFactor final : public EigenFactor {};

}  // namespace lp_solver::linalg
