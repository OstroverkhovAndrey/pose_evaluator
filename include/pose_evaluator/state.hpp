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
  Vec3 p = Vec3::Zero();                     // position in world frame
  Vec3 v = Vec3::Zero();                     // linear velocity in world frame
  Eigen::Quaterniond q{1.0, 0.0, 0.0, 0.0}; // orientation in world frame
  Vec3 w = Vec3::Zero();                     // angular velocity
};

}  // namespace pose_evaluator
