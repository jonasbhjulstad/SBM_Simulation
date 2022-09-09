#include "Feature_Model.hpp"

namespace FROLS::Features {

Vec Feature_Model::step(crVec &x, crVec &u) const {
  Vec x_next(x.rows());
  x_next.setZero();
  Mat X(1, x.rows() + u.rows());
  X << x.transpose(), u.transpose();
  for (int i = 0; i < features.size(); i++) {
    for (int j = 0; j < features[i].size(); j++) {
      x_next(i) +=
          features[i][j].theta * transform(X, features[i][j].index).value();
    }
  }
  return x_next;
}
Mat Feature_Model::simulate(crVec &x0, crMat &U, size_t Nt) const {
  Mat X(Nt + 1, x0.rows());
  X.row(0) = x0;
  for (int i = 0; i < Nt; i++) {
    X.row(i + 1) = step(X.row(i), U.row(i));
  }
  return X;
}
} // namespace FROLS::Features