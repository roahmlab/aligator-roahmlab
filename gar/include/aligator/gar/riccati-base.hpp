#pragma once

#include "riccati-impl.hpp"

namespace aligator {
namespace gar {

template <typename _Scalar> class RiccatiSolverBase {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using StageFactorType = StageFactor<Scalar>;
  std::vector<StageFactorType> datas;

  virtual bool backward(const Scalar mudyn, const Scalar mueq) = 0;

  virtual bool
  forward(std::vector<VectorXs> &xs, std::vector<VectorXs> &us,
          std::vector<VectorXs> &vs, std::vector<VectorXs> &lbdas,
          const std::optional<ConstVectorRef> &theta_ = std::nullopt) const = 0;

  virtual ~RiccatiSolverBase() = default;
};

} // namespace gar
} // namespace aligator
