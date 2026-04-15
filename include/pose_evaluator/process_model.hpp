#pragma once
#include "pose_evaluator/state.hpp"

namespace pose_evaluator
{

using ProcessNoiseVec = Eigen::Matrix<double, 6, 1>;
using ProcessNoiseCov = Eigen::Matrix<double, 6, 6>;

class IProcessModel
{
public:
  virtual ~IProcessModel() = default;

  virtual State propagate(
    const State & x,
    const ProcessNoiseVec & noise,
    double dt) const = 0;

  virtual ProcessNoiseCov noiseCov(double dt) const = 0;
};

}  // namespace pose_evaluator
