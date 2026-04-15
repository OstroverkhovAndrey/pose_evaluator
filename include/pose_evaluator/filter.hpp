
#pragma once
#include "pose_evaluator/state.hpp"
#include "pose_evaluator/measurement_model.hpp"

namespace pose_evaluator
{

class IFilter
{
public:
  virtual ~IFilter() = default;

  virtual void initialize(const State & x0, const Cov12 & P0) = 0;
  virtual bool isInitialized() const = 0;

  virtual void predict(double dt) = 0;
  virtual void update(const Eigen::VectorXd & z, const IMeasurementModel & model) = 0;

  virtual const State & state() const = 0;
  virtual const Cov12 & covariance() const = 0;
};

}  // namespace pose_evaluator
