/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
/// @author Wilson Jallet
#pragma once

#include "aligator/math.hpp"

#include <optional>

namespace aligator {
namespace gar {

template <typename _Scalar> class RiccatiSolverBase {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  virtual bool backward(const Scalar mudyn, const Scalar mueq) = 0;

  virtual bool
  forward(std::vector<VectorXs> &xs, std::vector<VectorXs> &us,
          std::vector<VectorXs> &vs, std::vector<VectorXs> &lbdas,
          const std::optional<ConstVectorRef> &theta_ = std::nullopt) const = 0;

  /// For applicable solvers, updates the first feedback gain in-place to
  /// correspond to the first Riccati gain.
  virtual void collapseFeedback() {}

  virtual ~RiccatiSolverBase() = default;
};

} // namespace gar
} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "riccati-base.txx"
#endif
