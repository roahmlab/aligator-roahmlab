/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
///
/// @page fddp_intro The FDDP algorithm
/// @htmlinclude fddp.html
#pragma once

#include "proxddp/core/solver-util.hpp"
#include "proxddp/core/helpers-base.hpp"
#include "proxddp/core/explicit-dynamics.hpp"

#include "proxddp/fddp/results.hpp"
#include "proxddp/fddp/workspace.hpp"
#include "proxddp/fddp/linesearch.hpp"

#include "proxddp/utils/logger.hpp"
#include "proxddp/utils/rollout.hpp"

#include <fmt/ostream.h>

#define PROXDDP_FDDP_WARNING(msg)                                              \
  fmt::print(fmt::fg(fmt::color::yellow), "[SolverFDDP] ({}) warning: {}",     \
             __FUNCTION__, msg)

namespace proxddp {

/**
 * @brief The feasible DDP (FDDP) algorithm
 *
 */
template <typename Scalar> struct SolverFDDP {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Problem = TrajOptProblemTpl<Scalar>;
  using StageModel = StageModelTpl<Scalar>;
  using StageData = StageDataTpl<Scalar>;
  using ProblemData = TrajOptDataTpl<Scalar>;
  using Results = ResultsFDDPTpl<Scalar>;
  using Workspace = WorkspaceFDDPTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using VParams = internal::value_storage<Scalar>;
  using QParams = internal::q_storage<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;
  using ExpModel = ExplicitDynamicsModelTpl<Scalar>;
  using ExpData = ExplicitDynamicsDataTpl<Scalar>;
  using CallbackPtr = shared_ptr<helpers::base_callback<Scalar>>;

  Scalar target_tol_;

  /// @name Regularization parameters
  /// \{
  Scalar reg_init;
  Scalar xreg_ = reg_init;
  Scalar ureg_ = xreg_;
  Scalar reg_min_ = 1e-9;
  Scalar reg_max_ = 1e9;
  /// Regularization decrease factor
  Scalar reg_dec_factor_ = 0.1;
  /// Regularization increase factor
  Scalar reg_inc_factor_ = 10.;
  /// \}

  Scalar th_grad_ = 1e-12;
  Scalar th_step_dec_ = 0.5;
  Scalar th_step_inc_ = 0.01;

  typename Linesearch<Scalar>::Options ls_params;

  VerboseLevel verbose_;
  /// Maximum number of iterations for the solver.
  std::size_t max_iters;

  BaseLogger logger{false};

  /// Callbacks
  std::vector<CallbackPtr> callbacks_;

  std::unique_ptr<Results> results_;
  std::unique_ptr<Workspace> workspace_;

  SolverFDDP(const Scalar tol = 1e-6,
             VerboseLevel verbose = VerboseLevel::QUIET,
             const Scalar reg_init = 1e-9, const std::size_t max_iters = 200);

  const Results &getResults() const { return *results_; }
  const Workspace &getWorkspace() const { return *workspace_; }

  /// @brief Allocate workspace and results structs.
  void setup(const Problem &problem);

  /**
   * @brief   Perform a nonlinear rollout, keeping an infeasibility gap.
   * @details Perform a nonlinear rollout using the computed sensitivity gains
   * from the backward pass, while keeping the dynamical feasibility gaps open
   * proportionally to the step-size @p alpha.
   * @param[in]   problem
   * @param[in]   results
   * @param[out]  workspace
   * @param[in]   alpha step-size.
   */
  static Scalar forwardPass(const Problem &problem, const Results &results,
                            Workspace &workspace, const Scalar alpha);

  /**
   * @brief     Pre-compute the improvement model params.
   * @details   Inspired from Crocoddyl.
   */
  void updateExpectedImprovement(Workspace &workspace, Results &results) const;

  /**
   * @brief    Correct the directional derivatives.
   * @details  This will re-compute the gap between the results trajectory and
   * the trial trajectory.
   */
  void expectedImprovement(Workspace &workspace, Scalar &d1, Scalar &d2) const;

  /**
   * @brief   Computes dynamical feasibility gaps.
   * @details This computes the difference \f$x_{i+1} \ominus f(x_i, u_i)$, as
   * well as the residual of initial condition. This function will compute the
   * forward dynamics at every step to compute the forward map $f(x_i, u_i)$.
   */
  static Scalar computeInfeasibility(const Problem &problem,
                                     const std::vector<VectorXs> &xs,
                                     Workspace &workspace);

  /// @brief   Perform the backward pass and compute Riccati gains.
  void backwardPass(const Problem &problem, Workspace &workspace,
                    Results &results) const;

  void increaseRegularization() {
    xreg_ *= reg_inc_factor_;
    xreg_ = std::min(xreg_, reg_max_);
    ureg_ = xreg_;
  }

  void decreaseRegularization() {
    xreg_ *= reg_dec_factor_;
    xreg_ = std::max(xreg_, reg_min_);
    ureg_ = xreg_;
  }

  /// @brief Compute the dual feasibility of the problem.
  Scalar computeCriterion(Workspace &workspace);

  /// @brief    Add a callback to the solver instance.
  void registerCallback(const CallbackPtr &cb) { callbacks_.push_back(cb); }

  /// @brief    Remove all callbacks from the instance.
  void clearCallbacks() { callbacks_.clear(); }

  void invokeCallbacks(Workspace &workspace, Results &results) {
    for (auto &cb : callbacks_) {
      cb->call(workspace, results);
    }
  }

  bool run(const Problem &problem, const std::vector<VectorXs> &xs_init = {},
           const std::vector<VectorXs> &us_init = {});

  static const ExpData &
  stage_get_dynamics_data(const StageDataTpl<Scalar> &sd) {
    const auto &dd = sd.dyn_data();
    return static_cast<const ExpData &>(dd);
  }
};

} // namespace proxddp

#include "proxddp/fddp/solver-fddp.hxx"
