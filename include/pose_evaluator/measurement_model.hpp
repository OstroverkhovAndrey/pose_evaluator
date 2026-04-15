#pragma once
#include "pose_evaluator/state.hpp"
#include <Eigen/Dense>

namespace pose_evaluator
{

class IMeasurementModel
{
public:
  virtual ~IMeasurementModel() = default;

  virtual int measurementDim() const = 0;
  virtual Eigen::VectorXd predictMeasurement(const State & x) const = 0;
  virtual Eigen::MatrixXd measurementCov() const = 0;
};

}  // namespace pose_evaluator
