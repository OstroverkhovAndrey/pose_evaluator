#pragma once
#include "pose_evaluator/state.hpp"
#include "pose_evaluator/so3_utils.hpp"

namespace pose_evaluator
{

struct StateOps
{
  static State boxPlus(const State & x, const ErrorVec & dx)
  {
    State out = x;

    // Позиция и скорости обновляются обычным сложением
    out.p += dx.segment<3>(0);
    out.v += dx.segment<3>(3);
    out.w += dx.segment<3>(9);

    // Левое приращение ориентации:
    // q_new = Exp(dtheta) * q
    const Eigen::Vector3d dtheta = dx.segment<3>(6);
    out.q = (quatExp(dtheta) * x.q).normalized();

    return out;
  }

  static ErrorVec boxMinus(const State & a, const State & b)
  {
    ErrorVec dx;

    dx.segment<3>(0) = a.p - b.p;
    dx.segment<3>(3) = a.v - b.v;
    dx.segment<3>(9) = a.w - b.w;

    // При левом приращении относительная ошибка ориентации должна быть:
    // dq = a.q * b.q^{-1}
    const Eigen::Quaterniond dq = a.q * b.q.conjugate();
    dx.segment<3>(6) = quatLog(dq);

    return dx;
  }
};

}  // namespace pose_evaluator
