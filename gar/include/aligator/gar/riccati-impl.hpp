/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "./lqr-problem.hpp"
#include "./blk-matrix.hpp"

#include <proxsuite-nlp/linalg/bunchkaufman.hpp>
#include <Eigen/Cholesky>

#include <boost/core/make_span.hpp>

namespace aligator {
namespace gar {

/// Per-node struct for all computations in the factorization.
template <typename Scalar> struct StageFactor {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using RowMatrixXs = Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>;

  struct value_t {
    MatrixXs Pmat;                  //< Riccati matrix
    VectorXs pvec;                  //< Riccati bias
    MatrixXs schurMat;              //< Dual-space Schur matrix
    MatrixXs Vxx;                   //< "cost-to-go" matrix
    VectorXs vx;                    //< "cost-to-go" gradient
    Eigen::LLT<MatrixXs> Pchol;     //< Cholesky decomposition of Pmat
    Eigen::LLT<MatrixXs> schurChol; //< Cholesky decomposition of Schur matrix
    MatrixXs Vxt;
    MatrixXs Vtt;
    VectorXs vt;

    value_t(uint nx, uint nth)
        : Pmat(nx, nx), pvec(nx), schurMat(nx, nx), //
          Vxx(nx, nx), vx(nx), Pchol(nx), schurChol(nx), Vxt(nx, nth),
          Vtt(nth, nth), vt(nth) {
      Vxt.setZero();
      Vtt.setZero();
      vt.setZero();
    }
  };

  StageFactor(uint nx, uint nu, uint nc, uint nth)
      : Qhat(nx, nx), Rhat(nu, nu), Shat(nx, nu), qhat(nx), rhat(nu),
        AtV(nx, nx), BtV(nu, nx), Gxhat(nx, nth), Guhat(nu, nth),
        ff({nu, nc, nx, nx}, {1}), fb({nu, nc, nx, nx}, {nx}),
        fth({nu, nc, nx, nx}, {nth}),                       //
        kktMat({nu, nc}, {nu, nc}), kktChol(kktMat.rows()), //
        vm(nx, nth), PinvEt(nx, nx) {
    Qhat.setZero();
    Rhat.setZero();
    Shat.setZero();
    qhat.setZero();
    rhat.setZero();

    AtV.setZero();
    BtV.setZero();

    Gxhat.setZero();
    Guhat.setZero();

    ff.setZero();
    fb.setZero();
    kktMat.setZero();
    fth.setZero();
  }

  MatrixXs Qhat;
  MatrixXs Rhat;
  MatrixXs Shat;
  VectorXs qhat;
  VectorXs rhat;
  RowMatrixXs AtV;
  RowMatrixXs BtV;

  // Parametric
  MatrixXs Gxhat;
  MatrixXs Guhat;

  BlkMatrix<VectorXs, 4, 1> ff;          //< feedforward gains
  BlkMatrix<RowMatrixXs, 4, 1> fb;       //< feedback gains
  BlkMatrix<RowMatrixXs, 4, 1> fth;      //< parameter feedback gains
  BlkMatrix<MatrixXs, 2, 2> kktMat;      //< reduced KKT matrix buffer
  Eigen::BunchKaufman<MatrixXs> kktChol; //< reduced KKT LDLT solver
  value_t vm;                            //< cost-to-go parameters
  MatrixXs PinvEt;                       //< tmp buffer for \f$P^{-1}E^\top\f$
};

/// A sequential, regularized Riccati algorithm
// for proximal-regularized, constrained LQ problems.
template <typename Scalar> struct ProximalRiccatiImpl {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using RowMatrixXs = Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>;
  using RowMatrixRef = Eigen::Ref<RowMatrixXs>;
  using ConstRowMatrixRef = Eigen::Ref<const RowMatrixXs>;
  using KnotType = LQRKnotTpl<Scalar>;
  using StageFactor = StageFactor<Scalar>;
  using value_t = typename StageFactor::value_t;

  struct kkt0_t {
    BlkMatrix<MatrixXs, 2, 2> mat;
    BlkMatrix<VectorXs, 2, 1> ff;
    BlkMatrix<RowMatrixXs, 2, 1> fth;
    Eigen::BunchKaufman<MatrixXs> chol{mat.rows()};
    kkt0_t(uint nx, uint nc, uint nth)
        : mat({nx, nc}, {nx, nc}), ff(mat.rowDims()),
          fth(mat.rowDims(), {nth}) {}
  };

  inline static void computeMatrixTerms(const KnotType &model, Scalar mudyn,
                                        Scalar mueq, value_t &vnext,
                                        StageFactor &d);

  inline static bool backwardImpl(boost::span<const KnotType> stages,
                                  const Scalar mudyn, const Scalar mueq,
                                  boost::span<StageFactor> datas);

  /// Solve initial stage
  inline static void
  computeInitial(VectorRef x0, VectorRef lbd0, const kkt0_t &kkt0,
                 const boost::optional<ConstVectorRef> &theta_) {
    assert(kkt0.chol.info() == Eigen::Success);
    x0 = kkt0.ff.blockSegment(0);
    lbd0 = kkt0.ff.blockSegment(1);
    if (theta_.has_value()) {
      x0.noalias() += kkt0.fth.blockRow(0) * theta_.value();
      lbd0.noalias() += kkt0.fth.blockRow(1) * theta_.value();
    }
  }

  inline static void solveOneStage(const KnotType &model, StageFactor &d,
                                   value_t &vn, const Scalar mudyn,
                                   const Scalar mueq);

  /// Forward sweep.
  inline static bool
  forwardImpl(boost::span<const KnotType> stages,
              boost::span<const StageFactor> datas, boost::span<VectorXs> xs,
              boost::span<VectorXs> us, boost::span<VectorXs> vs,
              boost::span<VectorXs> lbdas,
              const boost::optional<ConstVectorRef> &theta_ = boost::none);
};

} // namespace gar
} // namespace aligator

#include "./riccati-impl.hxx"
