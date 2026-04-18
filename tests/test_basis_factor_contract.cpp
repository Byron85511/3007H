#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "lp_solver/lp_solver.hpp"

using namespace lp_solver;

class BasisFactorContract : public ::testing::TestWithParam<linalg::FactorBackend> {
 protected:
  std::unique_ptr<linalg::IBasisFactor> factor() { return linalg::makeFactor(GetParam()); }
};

TEST_P(BasisFactorContract, FactorisesDiagonalMatrix) {
  auto B = util::PackedMatrix::Builder(3, 3)
               .appendColumn({0}, {2.0})
               .appendColumn({1}, {4.0})
               .appendColumn({2}, {5.0})
               .build();

  ASSERT_TRUE(factor()->factorize(B));
}

TEST_P(BasisFactorContract, FtranSolvesCorrectly) {
  auto B = util::PackedMatrix::Builder(3, 3)
               .appendColumn({0}, {2.0})
               .appendColumn({1}, {4.0})
               .appendColumn({2}, {5.0})
               .build();

  auto f = factor();
  ASSERT_TRUE(f->factorize(B));

  util::IndexedVector rhs(3);
  rhs.add(0, 2.0);
  rhs.add(1, 4.0);
  rhs.add(2, 5.0);
  f->ftran(rhs);

  EXPECT_NEAR(rhs[0], 1.0, 1e-10);
  EXPECT_NEAR(rhs[1], 1.0, 1e-10);
  EXPECT_NEAR(rhs[2], 1.0, 1e-10);
}

TEST_P(BasisFactorContract, BtranSolvesCorrectly) {
  auto B = util::PackedMatrix::Builder(3, 3)
               .appendColumn({0}, {2.0})
               .appendColumn({1}, {4.0})
               .appendColumn({2}, {5.0})
               .build();

  auto f = factor();
  ASSERT_TRUE(f->factorize(B));

  util::IndexedVector rhs(3);
  rhs.add(0, 2.0);
  rhs.add(1, 4.0);
  rhs.add(2, 5.0);
  f->btran(rhs);

  EXPECT_NEAR(rhs[0], 1.0, 1e-10);
  EXPECT_NEAR(rhs[1], 1.0, 1e-10);
  EXPECT_NEAR(rhs[2], 1.0, 1e-10);
}

TEST_P(BasisFactorContract, ReturnsFalseForSingularMatrix) {
  auto B = util::PackedMatrix::Builder(2, 2).appendColumn({}, {}).appendColumn({}, {}).build();

  EXPECT_FALSE(factor()->factorize(B));
}

INSTANTIATE_TEST_SUITE_P(AllBackends, BasisFactorContract,
                         ::testing::Values(linalg::FactorBackend::Eigen,
                                           linalg::FactorBackend::Umfpack),
                         [](const auto& info) {
                           return info.param == linalg::FactorBackend::Eigen ? "Eigen"
                                                                             : "Umfpack";
                         });

TEST_P(BasisFactorContract, FtranBtranDense3x3MatchesReference) {
  // Manual LU example matrix (MAT3007H Project Manual).
  auto B = util::PackedMatrix::Builder(3, 3)
               .appendColumn({0, 1, 2}, {2.0, 4.0, -2.0})
               .appendColumn({0, 1, 2}, {1.0, -6.0, 7.0})
               .appendColumn({0, 1, 2}, {1.0, 0.0, 2.0})
               .build();

  Eigen::Matrix3d Bd;
  Bd << 2, 1, 1, 4, -6, 0, -2, 7, 2;

  auto f = factor();
  ASSERT_TRUE(f->factorize(B));

  util::IndexedVector rhs(3);
  rhs.add(0, 4.0);
  rhs.add(1, -2.0);
  rhs.add(2, 7.0);
  f->ftran(rhs);

  Eigen::Vector3d x(rhs[0], rhs[1], rhs[2]);
  Eigen::Vector3d expected = Bd.partialPivLu().solve(Eigen::Vector3d(4, -2, 7));
  EXPECT_NEAR(x[0], expected[0], 1e-9);
  EXPECT_NEAR(x[1], expected[1], 1e-9);
  EXPECT_NEAR(x[2], expected[2], 1e-9);

  util::IndexedVector br(3);
  br.add(0, 1.0);
  br.add(1, 0.0);
  br.add(2, 0.0);
  f->btran(br);
  Eigen::Vector3d bt(br[0], br[1], br[2]);
  Eigen::Vector3d expected_bt = Bd.transpose().partialPivLu().solve(Eigen::Vector3d(1, 0, 0));
  EXPECT_NEAR(bt[0], expected_bt[0], 1e-9);
  EXPECT_NEAR(bt[1], expected_bt[1], 1e-9);
  EXPECT_NEAR(bt[2], expected_bt[2], 1e-9);
}

TEST_P(BasisFactorContract, EtaFileMatchesExplicitBasisUpdate) {
  auto B0 = util::PackedMatrix::Builder(3, 3)
                .appendColumn({0, 1, 2}, {2.0, 1.0, -1.0})
                .appendColumn({0, 1, 2}, {0.0, 2.0, 1.0})
                .appendColumn({0, 1, 2}, {1.0, 0.0, 1.0})
                .build();

  Eigen::Matrix3d B0d;
  B0d << 2, 0, 1, 1, 2, 0, -1, 1, 1;

  const int p = 1;
  Eigen::Vector3d aq(1.0, -1.0, 2.0);
  Eigen::Vector3d d = B0d.partialPivLu().solve(aq);

  auto f = factor();
  ASSERT_TRUE(f->factorize(B0));

  util::IndexedVector d_iv(3);
  for (int i = 0; i < 3; ++i) {
    d_iv.set(i, d[static_cast<int>(i)]);
  }
  f->updateEta(p, d_iv);

  util::IndexedVector rhs(3);
  rhs.add(0, 3.0);
  rhs.add(1, -1.0);
  rhs.add(2, 2.0);
  f->ftran(rhs);

  Eigen::Matrix3d E = Eigen::Matrix3d::Identity();
  E.col(p) = d;
  Eigen::Matrix3d B1d = B0d * E;
  Eigen::Vector3d rhs_dense(3, -1, 2);
  Eigen::Vector3d expected = B1d.partialPivLu().solve(rhs_dense);

  EXPECT_NEAR(rhs[0], expected[0], 1e-8);
  EXPECT_NEAR(rhs[1], expected[1], 1e-8);
  EXPECT_NEAR(rhs[2], expected[2], 1e-8);
}
