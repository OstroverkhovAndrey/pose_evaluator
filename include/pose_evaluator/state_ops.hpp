
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
    out.p += dx.segment<3>(0);
    out.v += dx.segment<3>(3);
    out.q = (x.q * quatExp(dx.segment<3>(6))).normalized();
    out.w += dx.segment<3>(9);
    return out;
  }

  static ErrorVec boxMinus(const State & a, const State & b)
  {
    ErrorVec dx;
    dx.segment<3>(0) = a.p - b.p;
    dx.segment<3>(3) = a.v - b.v;
    dx.segment<3>(6) = quatLog(b.q.conjugate() * a.q);
    dx.segment<3>(9) = a.w - b.w;
    return dx;
  }
};

}  // namespace pose_evaluator
