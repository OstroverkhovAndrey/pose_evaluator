#pragma once
#include "pose_evaluator/process_model.hpp"
#include "pose_evaluator/so3_utils.hpp"

namespace pose_evaluator
{

class WhiteNoiseRigidBodyModel : public IProcessModel
{
public:
  WhiteNoiseRigidBodyModel(double sigma_a, double sigma_alpha)
  : sigma_a_(sigma_a), sigma_alpha_(sigma_alpha) {}

  State propagate(
    const State & x,
    const ProcessNoiseVec & noise,
    double dt) const override
  {
    const Eigen::Vector3d a = noise.segment<3>(0);
    const Eigen::Vector3d alpha = noise.segment<3>(3);

    State out;
    out.p = x.p + x.v * dt + 0.5 * a * dt * dt;
    out.v = x.v + a * dt;

    const Eigen::Vector3d dtheta = x.w * dt + 0.5 * alpha * dt * dt;
    out.q = (x.q * quatExp(dtheta)).normalized();

    out.w = x.w + alpha * dt;
    return out;
  }

  ProcessNoiseCov noiseCov(double /*dt*/) const override
  {
    ProcessNoiseCov Q = ProcessNoiseCov::Zero();
    Q.block<3,3>(0,0) = sigma_a_ * sigma_a_ * Eigen::Matrix3d::Identity();
    Q.block<3,3>(3,3) = sigma_alpha_ * sigma_alpha_ * Eigen::Matrix3d::Identity();
    return Q;
  }

private:
  double sigma_a_;
  double sigma_alpha_;
};

}  // namespace pose_evaluator
