#pragma once
#include "pose_evaluator/process_model.hpp"
#include "pose_evaluator/so3_utils.hpp"

namespace pose_evaluator
{

class RandomAccelProcessModel : public IProcessModel
{
public:
  RandomAccelProcessModel() = default;

  // a_alpha: [a_x, a_y, a_z, alpha_x, alpha_y, alpha_z]
  State propagate(const State& x, const ProcessNoiseVec& a_alpha, double dt) const override
  {
    Eigen::Vector3d a     = a_alpha.segment<3>(0);   // линейное ускорение
    Eigen::Vector3d alpha = a_alpha.segment<3>(3);   // угловое ускорение

    State out;
    out.p = x.p + x.v * dt + 0.5 * a * dt * dt;
    out.v = x.v + a * dt;

    Eigen::Vector3d dtheta = x.w * dt + 0.5 * alpha * dt * dt;
    out.q = (x.q * quatExp(dtheta)).normalized();

    out.w = x.w + alpha * dt;
    return out;
  }

  ProcessNoiseCov noiseCov(double /*dt*/) const override
  {
    // Не используется — шум явно подается в propagate
    return ProcessNoiseCov::Zero();
  }
};

} // namespace pose_evaluator
