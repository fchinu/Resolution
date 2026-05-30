#include <cmath>

// Define in C++ so that we compile it and run faster in python

double dscb_core(double x, double mu, double width,
                 double a1, double p1, double a2, double p2) {
  const double u  = (x - mu) / width;
  const double A1 = std::pow(p1 / std::fabs(a1), p1) * std::exp(-a1 * a1 / 2);
  const double A2 = std::pow(p2 / std::fabs(a2), p2) * std::exp(-a2 * a2 / 2);
  const double B1 = p1 / std::fabs(a1) - std::fabs(a1);
  const double B2 = p2 / std::fabs(a2) - std::fabs(a2);

  double result = 1.0;
  if      (u < -a1) result *= A1 * std::pow(B1 - u, -p1);
  else if (u <  a2) result *= std::exp(-u * u / 2);
  else              result *= A2 * std::pow(B2 + u, -p2);
  return result;
}

double double_sided_cb(double *x, double *par) {
  return par[0] * dscb_core(x[0], par[1], par[2], par[3], par[4], par[5], par[6]);
}

double double_sided_cb_plus_bkg(double *x, double *par) {
  return par[0] * dscb_core(x[0], par[1], par[2], par[3], par[4], par[5], par[6])
       + par[7] * std::exp(par[8] * x[0]);
}
