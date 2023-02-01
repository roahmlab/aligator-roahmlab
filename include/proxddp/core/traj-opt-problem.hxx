#pragma once

#include "proxddp/core/traj-opt-problem.hpp"
#include "proxddp/utils/exceptions.hpp"

#include <fmt/format.h>

namespace proxddp {

template <typename Scalar>
TrajOptProblemTpl<Scalar>::TrajOptProblemTpl(
    const VectorXs &x0, const std::vector<shared_ptr<StageModel>> &stages,
    const shared_ptr<CostAbstract> &term_cost)
    : init_state_error_(stages[0]->xspace_, stages[0]->nu(), x0),
      stages_(stages), term_cost_(term_cost), dummy_term_u0(stages[0]->nu()) {
  dummy_term_u0.setZero();
}

template <typename Scalar>
TrajOptProblemTpl<Scalar>::TrajOptProblemTpl(
    const VectorXs &x0, const int nu, const shared_ptr<Manifold> &space,
    const shared_ptr<CostAbstract> &term_cost)
    : TrajOptProblemTpl(StateErrorResidual(space, nu, x0), nu, term_cost) {}

template <typename Scalar>
Scalar TrajOptProblemTpl<Scalar>::evaluate(const std::vector<VectorXs> &xs,
                                           const std::vector<VectorXs> &us,
                                           Data &prob_data) const {
  const std::size_t nsteps = numSteps();
  const bool sizes_correct = (xs.size() == nsteps + 1) && (us.size() == nsteps);
  if (!sizes_correct) {
    PROXDDP_RUNTIME_ERROR(fmt::format(
        "Wrong size for xs or us, expected us.size = {:d}", nsteps));
  }

  init_state_error_.evaluate(xs[0], us[0], xs[1], prob_data.getInitData());

  for (std::size_t i = 0; i < nsteps; i++) {
    stages_[i]->evaluate(xs[i], us[i], xs[i + 1], prob_data.getStageData(i));
  }

  term_cost_->evaluate(xs[nsteps], dummy_term_u0, *prob_data.term_cost_data);

  for (std::size_t k = 0; k < term_cstrs_.size(); ++k) {
    const ConstraintType &tc = term_cstrs_[k];
    auto &td = prob_data.term_cstr_data[k];
    tc.func->evaluate(xs[nsteps], dummy_term_u0, xs[nsteps], *td);
  }
  prob_data.cost_ = computeTrajectoryCost(prob_data);
  return prob_data.cost_;
}

template <typename Scalar>
void TrajOptProblemTpl<Scalar>::computeDerivatives(
    const std::vector<VectorXs> &xs, const std::vector<VectorXs> &us,
    Data &prob_data) const {
  const std::size_t nsteps = numSteps();
  const bool sizes_correct = (xs.size() == nsteps + 1) && (us.size() == nsteps);
  if (!sizes_correct) {
    PROXDDP_RUNTIME_ERROR(fmt::format(
        "Wrong size for xs or us, expected us.size = {:d}", nsteps));
  }

  init_state_error_.computeJacobians(xs[0], us[0], xs[1],
                                     prob_data.getInitData());

  for (std::size_t i = 0; i < nsteps; i++) {
    stages_[i]->computeDerivatives(xs[i], us[i], xs[i + 1],
                                   prob_data.getStageData(i));
  }

  if (term_cost_) {
    term_cost_->computeGradients(xs[nsteps], dummy_term_u0,
                                 *prob_data.term_cost_data);
    term_cost_->computeHessians(xs[nsteps], dummy_term_u0,
                                *prob_data.term_cost_data);
  }

  for (std::size_t k = 0; k < term_cstrs_.size(); ++k) {
    const ConstraintType &tc = term_cstrs_[k];
    auto &td = prob_data.term_cstr_data[k];
    tc.func->computeJacobians(xs[nsteps], dummy_term_u0, xs[nsteps], *td);
  }
}

template <typename Scalar>
void TrajOptProblemTpl<Scalar>::addStage(const shared_ptr<StageModel> &stage) {
  stages_.push_back(stage);
}

template <typename Scalar>
void TrajOptProblemTpl<Scalar>::setTerminalConstraint(
    const ConstraintType &cstr) {
  removeTerminalConstraints();
  addTerminalConstraint(cstr);
}

template <typename Scalar>
void TrajOptProblemTpl<Scalar>::addTerminalConstraint(
    const ConstraintType &cstr) {
  term_cstrs_.pushBack(cstr);
}

template <typename Scalar>
inline std::size_t TrajOptProblemTpl<Scalar>::numSteps() const {
  return stages_.size();
}

template <typename Scalar>
void TrajOptProblemTpl<Scalar>::replaceStageCircular(
    const shared_ptr<StageModel> &model) {
  addStage(model);
  rotate_vec_left(stages_);
  stages_.pop_back();
}

template <typename Scalar>
Scalar TrajOptProblemTpl<Scalar>::computeTrajectoryCost(
    const Data &problem_data) const {
  PROXDDP_NOMALLOC_BEGIN;
  Scalar traj_cost = 0.;

  const std::size_t nsteps = numSteps();
  for (std::size_t step = 0; step < nsteps; step++) {
    const StageDataTpl<Scalar> &sd = problem_data.getStageData(step);
    traj_cost += sd.cost_data->value_;
  }
  traj_cost += problem_data.term_cost_data->value_;

  PROXDDP_NOMALLOC_END;
  return traj_cost;
}

/* TrajOptDataTpl */

template <typename Scalar>
TrajOptDataTpl<Scalar>::TrajOptDataTpl(const TrajOptProblemTpl<Scalar> &problem)
    : init_data(problem.init_state_error_.createData()) {
  stage_data.reserve(problem.numSteps());
  for (std::size_t i = 0; i < problem.numSteps(); i++) {
    stage_data.push_back(problem.stages_[i]->createData());
    stage_data[i]->checkData();
  }

  if (problem.term_cost_) {
    term_cost_data = problem.term_cost_->createData();
  }

  if (!problem.term_cstrs_.empty())
    term_cstr_data.reserve(problem.term_cstrs_.size());
  for (std::size_t k = 0; k < problem.term_cstrs_.size(); k++) {
    const ConstraintType &tc = problem.term_cstrs_[k];
    term_cstr_data.push_back(tc.func->createData());
  }
}

} // namespace proxddp
