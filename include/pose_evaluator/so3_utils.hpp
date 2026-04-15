#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>

namespace pose_evaluator
{

inline Eigen::Quaterniond quatExp(const Eigen::Vector3d & dtheta)
{
  double angle = dtheta.norm();
  if (angle < 1e-12) {
    return Eigen::Quaterniond(
      1.0,
      0.5 * dtheta.x(),
      0.5 * dtheta.y(),
      0.5 * dtheta.z()).normalized();
  }

  Eigen::Vector3d axis = dtheta / angle;
  double half = 0.5 * angle;
  return Eigen::Quaterniond(
    std::cos(half),
    axis.x() * std::sin(half),
    axis.y() * std::sin(half),
    axis.z() * std::sin(half));
}

inline Eigen::Vector3d quatLog(const Eigen::Quaterniond & q_in)
{
  Eigen::Quaterniond q = q_in.normalized();
  if (q.w() < 0.0) {
    q.coeffs() *= -1.0;
  }

  Eigen::Vector3d v(q.x(), q.y(), q.z());
  double n = v.norm();

  if (n < 1e-12) {
    return 2.0 * v;
  }

  double angle = 2.0 * std::atan2(n, q.w());
  return angle * v / n;
}

}  // namespace pose_evaluator
