
#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace pose_evaluator
{

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

using ErrorVec = Eigen::Matrix<double, 12, 1>;
using Cov12 = Eigen::Matrix<double, 12, 12>;

struct State
{
  Vec3 p = Vec3::Zero();   // world position
  Vec3 v = Vec3::Zero();   // world linear velocity
  Eigen::Quaterniond q{1.0, 0.0, 0.0, 0.0}; // object orientation in world
  Vec3 w = Vec3::Zero();   // angular velocity in object/world approximation
};

}  // namespace pose_evaluator
