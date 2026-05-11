#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace pose_evaluator
{

using Vec3 = Eigen::Vector3d;
using Cov12 = Eigen::Matrix<double, 12, 12>;
using ErrorVec = Eigen::Matrix<double, 12, 1>;

struct State
{
  Vec3 p = Vec3::Zero();
  Vec3 v = Vec3::Zero();
  Eigen::Quaterniond q{1.0, 0.0, 0.0, 0.0};
  Vec3 w = Vec3::Zero();
};

}  // namespace pose_evaluator
