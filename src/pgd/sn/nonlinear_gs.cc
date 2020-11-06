#include "pgd/sn/nonlinear_gs.h"

namespace aether::pgd::sn {

NonlinearGS::NonlinearGS(std::vector<LinearInterface*> &linear_ops,
                         int num_materials, int num_legendre, int num_sources)
    : linear_ops(linear_ops),
      inner_products_x(linear_ops.size()),
      inner_products_b(linear_ops.size(), std::vector<double>(num_sources)),
      inner_products_all_x(linear_ops.size()),
      inner_products_all_b(linear_ops.size()),
      inner_products_one(num_materials, num_legendre) {
  inner_products_one = 1;
}

double NonlinearGS::step(dealii::BlockVector<double> x, 
                       const dealii::BlockVector<double> b,
                       const bool should_normalize,
                       const bool should_line_search) {
  std::vector<InnerProducts> coefficients_x(inner_products_x[0].size(),
                                            inner_products_x[0].back());
  std::vector<double> coefficients_b(inner_products_b[0].size());
  std::vector<dealii::Vector<double>> steps(linear_ops.size());
  std::vector<double> norms(linear_ops.size());
  /*
  double r0 = get_residual();
  const int num_modes = inner_products_x[0].size();
  const int num_sources = inner_products_b[0].size();
  std::vector<std::vector<InnerProducts>> ip_x_mode_mode(inner_products_x);
  std::vector<std::vector<InnerProducts>> ip_x_step_mode(inner_products_x);
  std::vector<InnerProducts> ip_x_step_step(
      linear_ops.size(), inner_products_x[0].back());
  std::vector<InnerProducts> ip_x_mode_step(
      linear_ops.size(), inner_products_x[0].back());
  std::vector<std::vector<double>> ip_b_mode(inner_products_b);
  std::vector<std::vector<double>> ip_b_step(inner_products_b);
  // for uniform weighting?
  std::vector<std::vector<InnerProducts>> ip_x_ones_mode(inner_products_x);
  std::vector<InnerProducts> ip_x_ones_step(ip_x_step_step);
  std::vector<std::vector<double>> ip_b_ones(inner_products_b);
  */
  double res = 0;
  linear_ops.back()->normalize();
  linear_ops.back()->get_inner_products(inner_products_x.back(), 
                                        inner_products_b.back());
  for (int i = 0; i < linear_ops.size(); ++i) {
    set_coefficients(i, coefficients_x, coefficients_b);
    res += std::pow(
        linear_ops[i]->get_residual(coefficients_x, coefficients_b), 2);
    linear_ops[i]->step(steps[i], b, coefficients_x, coefficients_b);
    // if (should_normalize) {
    //   norms[i] = linear_ops[i]->normalize();
    //   if (i == 1) {
    //     steps[i] /= norms[i];
    //     steps[i-1] *= norms[i];
    //     linear_ops[i-1]->scale(norms[i]);
    //     linear_ops[i-1]->get_inner_products(inner_products_x[i-1], 
    //                                         inner_products_b[i-1]);
    //   }
    // }

    // steps[i] = 1;
    // linear_ops[i]->get_inner_products_x(
    //     ip_x_mode_mode[i], steps[i]);
    // linear_ops[i]->get_inner_products_b(
    //     ip_b_mode[i], steps[i]);

    // linear_ops[i]->get_inner_products_x(
    //     ip_x_step_mode[i], steps[i]);
    // linear_ops[i]->get_inner_products_x(
    //     ip_x_mode_step[i], steps[i]);
    // linear_ops[i]->get_inner_products_x(
    //     ip_x_step_step[i], steps[i], steps[i]);
    // linear_ops[i]->get_inner_products_b(
    //     ip_b_step[i], steps[i]);

    // dealii::Vector<double> ones(steps[i]);
    // ones = 1;
    // linear_ops[i]->get_inner_products_x(
    //     ip_x_ones_mode[i], ones);
    // linear_ops[i]->get_inner_products_x(
    //     ip_x_ones_step[i], ones, steps[i]);
    // linear_ops[i]->get_inner_products_b(
    //     ip_b_ones[i], ones);

    linear_ops[i]->take_step(1.0, steps[i]);
    if (i < linear_ops.size() - 1)
      linear_ops[i]->get_inner_products(inner_products_x[i],
                                        inner_products_b[i]);
  }
  // // full-order residual line search?
  // // residual line search?
  // double lambda = 1;
  // for (int i = 0; i < linear_ops.size(); ++i) {
  //   linear_ops[i]->take_step(-1, steps[i]);
  //   linear_ops[i]->get_inner_products(inner_products_x[i], inner_products_b[i]);
  // }
  // // lambda = line_search(steps);  // this is different from what follows!
  // std::vector<double> c6(7);
  // linear_ops[0]->line_search(
  //     c6, steps[0],
  //     ip_x_mode_step[1], ip_x_step_step[1],
  //     ip_x_mode_mode[1], ip_x_step_mode[1],
  //     ip_b_mode[1], ip_b_step[1]);
  // // linear_ops[1]->line_search(
  // //     c6, steps[1],
  // //     ip_x_mode_step[0], ip_x_step_step[0],
  // //     ip_x_mode_mode[0], ip_x_step_mode[0],
  // //     ip_b_mode[0], ip_b_step[0]);
  // // lambda = lambda > 0.05 ? lambda : 1;
  // for (int i = 0; i < linear_ops.size(); ++i) {
  //   linear_ops[i]->take_step(+1, steps[i]);
  //   linear_ops[i]->get_inner_products(inner_products_x[i], inner_products_b[i]);
  // }
  // {
  //   auto res = [](std::vector<double> c, std::complex<double> a) {
  //     std::complex<double> r = c[0];
  //     for (int i = 1; i < c.size(); ++i)
  //       r += c[i] * std::pow(a, i);
  //     return r;
  //   };
  //   std::vector<double> dc(c6.size()-1);
  //   for (int i = 0; i < dc.size(); ++i)
  //     dc[i] = (i+1) * c6[i+1];
  //   std::vector<double> ddc(dc.size()-1);
  //   for (int i = 0; i < ddc.size(); ++i)
  //     ddc[i] = (i+1) * dc[i+1];
  //   std::vector<std::complex<double>> z(dc.size());
  //   for (int i = 0; i < z.size(); ++i)
  //     z[i] = std::pow(std::complex<double>(0.4, 0.9), i);
  //   auto p = std::bind(res, dc, std::placeholders::_1);
  //   auto pp = std::bind(res, ddc, std::placeholders::_1);
  //   auto aberth = [p, pp](std::vector<std::complex<double>> &z) {
  //     std::complex<double> w;
  //     for (int n = 0; n < 50; ++n) {
  //       for (int i = 0; i < z.size(); ++i) {
  //         std::complex<double> ratio = p(z[i]) / pp(z[i]);
  //         std::complex<double> sum = 0;
  //         for (int j = 0; j < z.size(); ++j)
  //           if (i != j)
  //             sum += 1. / (z[i] - z[j]);
  //         w = ratio / (1. - ratio * sum);
  //         if (std::isfinite(std::abs(w)))
  //           z[i] -= w;
  //       }
  //     }
  //   };
  //   aberth(z);
  //   double res_lambda = std::numeric_limits<double>::infinity(); //std::abs(res(c, lambda));
  //   for (int i = 0; i < z.size(); ++i) {
  //     // std::cout << z[i] << " ";
  //     if (z[i].imag() < 1e-16) {  // && std::abs(z[i].real()) > 5e-2
  //       // if (std::abs(z[i].real()) < 1e-2)
  //       //   continue;
  //       double res_zi = std::abs(res(c6, z[i].real()));
  //       if (res_zi < res_lambda) {
  //         lambda = z[i].real();
  //         res_lambda = res_zi;
  //         std::cout << lambda << " ";
  //       }
  //     }
  //   }
  //   std::cout << std::endl;
  // }
  // // L1 line search?
  // std::vector<double> cu(3);
  // for (int b = 0; b < num_sources; ++b)
  //   cu[0] += ip_b_ones[0][b] * ip_b_ones[1][b];
  // for (int m = 0; m < num_modes; ++m)
  //   cu[0] -= (ip_x_ones_mode[0][m] * ip_x_ones_mode[1][m]).eval();
  // cu[1] -= (ip_x_ones_step[0] * ip_x_ones_mode[1][num_modes-1]).eval()
  //          + (ip_x_ones_mode[0][num_modes-1] * ip_x_ones_step[1]).eval();
  // cu[2] -= (ip_x_ones_step[0] * ip_x_ones_step[1]).eval();
  // // std::cout << "L1 " << cu[0] << std::endl;
  // // line search
  // std::vector<double> c(linear_ops.size()*2+1);
  // AssertThrow(linear_ops.size() == 2, dealii::ExcNotImplemented());
  // std::vector<double> f_mode_mode(num_sources, 1);
  // std::vector<double> f_step_mode(num_sources, 1);
  // std::vector<double> f_mode_step(num_sources, 1);
  // std::vector<double> f_step_step(num_sources, 1);
  // std::vector<double> a_mode_mode_mode_mode(num_modes, 1);
  // std::vector<double> a_step_mode_mode_mode(num_modes, 1);
  // std::vector<double> a_mode_step_mode_mode(num_modes, 1);
  // std::vector<double> a_step_step_mode_mode(num_modes, 1);
  // for (int b = 0; b < num_sources; ++b) {
  //   f_mode_mode[b] = ip_b_mode[0][b] * ip_b_mode[1][b];
  //   f_step_mode[b] = ip_b_step[0][b] * ip_b_mode[1][b];
  //   f_mode_step[b] = ip_b_mode[0][b] * ip_b_step[1][b];
  //   f_step_step[b] = ip_b_step[0][b] * ip_b_step[1][b];
  //   c[0] += f_mode_mode[b];
  //   c[1] += f_step_mode[b] + f_mode_step[b];
  //   c[2] += f_step_step[b];
  // }
  // double f0 = c[0];
  // for (int m = 0; m < num_modes; ++m) {
  //   a_mode_mode_mode_mode[m] = (ip_x_mode_mode[0][m] * ip_x_mode_mode[1][m]).eval();
  //   a_step_mode_mode_mode[m] = (ip_x_step_mode[0][m] * ip_x_mode_mode[1][m]).eval();
  //   a_mode_step_mode_mode[m] = (ip_x_mode_mode[0][m] * ip_x_step_mode[1][m]).eval();
  //   a_step_step_mode_mode[m] = (ip_x_step_mode[0][m] * ip_x_step_mode[1][m]).eval();
  //   c[0] -= a_mode_mode_mode_mode[m];
  //   c[1] -= a_step_mode_mode_mode[m] + a_mode_step_mode_mode[m];
  //   c[2] -= a_step_step_mode_mode[m];
  // }
  // double a0 = c[0] - f0;
  // // 1st
  // double a_mode_mode_step_mode = (ip_x_mode_step[0] * ip_x_mode_mode[1][num_modes-1]).eval();
  // double a_mode_mode_mode_step = (ip_x_mode_mode[0][num_modes-1] * ip_x_mode_step[1]).eval();
  // c[1] -= a_mode_mode_step_mode + a_mode_mode_mode_step;
  // // 2nd
  // double a_mode_mode_step_step = (ip_x_mode_step[0] * ip_x_mode_step[1]).eval();
  // double a_step_mode_step_mode = (ip_x_step_step[0] * ip_x_mode_mode[1][num_modes-1]).eval();
  // double a_mode_step_mode_step = (ip_x_mode_mode[0][num_modes-1] * ip_x_step_step[1]).eval();
  // double a_step_mode_mode_step = (ip_x_step_mode[0][num_modes-1] * ip_x_mode_step[1]).eval();
  // double a_mode_step_step_mode = (ip_x_mode_step[0] * ip_x_step_mode[1][num_modes-1]).eval();
  // c[2] -= a_mode_mode_step_step + a_step_mode_step_mode + a_mode_step_mode_step
  //         + a_step_mode_mode_step + a_mode_step_step_mode;
  // // 3rd
  // double a_mode_step_step_step = (ip_x_mode_step[0] * ip_x_step_step[1]).eval();
  // double a_step_mode_step_step = (ip_x_step_step[0] * ip_x_mode_step[1]).eval();
  // double a_step_step_mode_step = (ip_x_step_mode[0][num_modes-1] * ip_x_step_step[1]).eval();
  // double a_step_step_step_mode = (ip_x_step_step[0] * ip_x_step_mode[1][num_modes-1]).eval();
  // c[3] -= a_mode_step_step_step + a_step_mode_step_step 
  //         + a_step_step_mode_step + a_step_step_step_mode;
  // // 4th
  // double a_step_step_step_step = (ip_x_step_step[0] * ip_x_step_step[1]).eval();
  // c[4] -= a_step_step_step_step;
  // double r00 = c[0] / (c[0]-a_mode_mode_mode_mode[num_modes-1]);
  // r00 = std::abs(r00);
  // r00 = std::sqrt(r00);
  // r0 = r00;
  // // r0 = c[0] / f0;
  // double r1 = (c[4] + c[3] + c[2] + c[1] + c[0]) / (c[0]-a_mode_mode_mode_mode[num_modes-1]);
  // r0 = std::sqrt(std::abs(r1));
  // // std::cout << c[0] << " -> " << (c[4]+c[3]+c[2]+c[1]+c[0]) << std::endl;
  // std::function<double(double)> res = [c](double a) {
  //   return c[4]*std::pow(a, 4) + c[3]*std::pow(a, 3) + c[2]*std::pow(a, 2)
  //          + c[1] * a + c[0];
  // };
  // // std::cout << res(0) << " -> " << res(1) << std::endl;
  // // root the cubic
  // std::vector<double> dc(c.size()-1);
  // for (int i = 0; i < dc.size(); ++i)
  //   dc[i] = (i+1) * c[i+1];
  // double delta0 = std::pow(dc[2], 2) - 3 * dc[3] * dc[1];
  // double delta1 = 2 * std::pow(dc[2], 3) 
  //                 - 9 * dc[3] * dc[2] * dc[1] 
  //                 + 27 * std::pow(dc[3], 2) * dc[0];
  // double sqrt = std::sqrt(std::pow(delta1, 2) - 4*std::pow(delta0, 3));
  // double cbrt = std::cbrt((delta1+sqrt)/2);
  // std::complex<double> u0(1, 0);
  // std::complex<double> u1(-1./2, +std::cbrt(3)/2);
  // std::complex<double> u2 = u1 * u1; //u2(-1./2, -std::cbrt(3)/2);
  // std::vector<std::complex<double>> u = {u0, u1, u2};
  // std::vector<std::complex<double>> roots(3);
  // for (int i = 0; i < u.size(); ++i) {
  //   roots[i] = -(1./(3*dc[3])) * (dc[2] + u[i]*cbrt + delta0/(u[i]*cbrt));
  //   // std::cout << roots[i] << " ";
  // }
  // // std::cout << std::endl;
  // for (int i = 0; i < roots.size(); ++i) {
  //   // std::cout << res(roots[i].real()) << " ";
  // }
  // // std::cout << std::endl;
  // // root the quartic?
  // double p = (8*c[4]*c[2] - 3*std::pow(c[3], 2)) / (8*std::pow(c[4], 2));
  // double q = (std::pow(c[3], 3) - 4*c[4]*c[3]*c[2] + 8*std::pow(c[4], 2)*c[1])
  //            / (8*std::pow(c[4], 3));
  // delta0 = std::pow(c[2], 2) - 3*c[3]*c[1] + 12*c[4]*c[0];
  // delta1 = 2*std::pow(c[2], 3) - 9*c[3]*c[2]*c[1] + 27*std::pow(c[3], 2)*c[0]
  //          + 27*c[4]*std::pow(c[1], 2) - 72*c[4]*c[2]*c[0];
  // sqrt = std::sqrt(std::pow(delta1, 2) - 4*std::pow(delta0, 3));
  // std::complex<double> Q = std::cbrt((delta1+sqrt)/2);
  // std::complex<double> s = 0.5 * std::sqrt(-2/3 * p + 1/(3*c[4]) * (Q+delta0/Q));
  // std::vector<std::complex<double>> roots4(4);
  // std::complex<double> pm = 0.5 * std::sqrt(-4.*std::pow(s, 2) - 2*p + q/s);
  // double x0 = -c[3]/(4*c[4]);
  // roots4[0] = x0 - s + pm;
  // roots4[1] = x0 - s - pm;
  // roots4[2] = x0 + s + pm;
  // roots4[3] = x0 + s - pm;
  // for (auto &r : roots4) {
  //   // std::cout << r << " ";
  // }
  // // std::cout << std::endl;
  // // take the step
  // // double lambda = roots[0].real();
  // if (should_line_search) {
  //   // lambda = std::clamp(lambda, 0.1 , 1.3);
  //   // lambda = 0.5;
  //   for (int i = 0; i < linear_ops.size(); ++i) {
  //     linear_ops[i]->take_step(lambda-1.0, steps[i]);
  //     linear_ops[i]->get_inner_products(inner_products_x[i], inner_products_b[i]);
  //   }
  // }
  // // std::cout << r00 << " " << c[0] << " " << a0 << " " << f0 << std::endl;
  // // a_step_mode_mode_step = (ip_x_step)
  // // a[0b0001] = ip_x[0b00] * ip_x[0b01];
  // /*
  // double r1 = 0;
  // double rh = 0;
  // if (false) {
  //   std::vector<dealii::Vector<double>> steps1(linear_ops.size());
  //   for (int i = 0; i < linear_ops.size(); ++i) {
  //     set_coefficients(i, coefficients_x, coefficients_b);
  //     linear_ops[i]->step(steps1[i], b, coefficients_x, coefficients_b);
  //     linear_ops[i]->take_step(1.0, steps1[i]);
  //     linear_ops[i]->get_inner_products(inner_products_x[i], inner_products_b[i]);
  //   }
  //   for (int i = 0; i < linear_ops.size(); ++i) {
  //     linear_ops[i]->take_step(-1.0, steps1[i]);
  //     linear_ops[i]->take_step(-0.5, steps[i]);
  //     linear_ops[i]->get_inner_products(inner_products_x[i], inner_products_b[i]);
  //   }
  //   std::vector<dealii::Vector<double>> steps_h(linear_ops.size());
  //   for (int i = 0; i < linear_ops.size(); ++i) {
  //     set_coefficients(i, coefficients_x, coefficients_b);
  //     linear_ops[i]->step(steps_h[i], b, coefficients_x, coefficients_b);
  //     linear_ops[i]->take_step(1.0, steps_h[i]);
  //     linear_ops[i]->get_inner_products(inner_products_x[i], inner_products_b[i]);
  //   }
  //   r0 = 0;
  //   for (int i = 0; i < linear_ops.size(); ++i) {
  //     linear_ops[i]->take_step(-1.0, steps_h[i]);
  //     r0 += std::pow(steps[i].l2_norm()/norms[i], 2);
  //     rh += std::pow(steps_h[i].l2_norm()/norms[i], 2);
  //     r1 += std::pow(steps1[i].l2_norm()/norms[i], 2);
  //   }
  //   r0 = std::sqrt(r0);
  //   rh = std::sqrt(rh);
  //   r1 = std::sqrt(r1);
  // } else {
  //   r1 = get_residual();
  //   for (int i = 0; i < linear_ops.size(); ++i) {
  //     linear_ops[i]->take_step(-0.5, steps[i]);
  //     linear_ops[i]->get_inner_products(inner_products_x[i], inner_products_b[i]);
  //   }
  //   rh = get_residual();
  // }
  // double dr1 = (3*r1 - 4*rh + r0) / (1 - 0);
  // double dr0 = (r1 - 4*rh + 3*r0) / (1 - 0); 
  // double relaxation = 1 - (dr1 * (1 - 0)) / (dr1 - dr0);
  // std::cout << "relaxation : " << relaxation << std::endl;
  // relaxation = std::isfinite(relaxation) ? relaxation : 0.5;
  // relaxation = std::clamp(relaxation, 0.2, 0.9);
  // // return r0;
  // // relaxation = 0.5;
  // // relaxation = 1;
  // */
  // /*
  // double relaxation = 0;
  // for (int i = 0; i < linear_ops.size(); ++i) {
  //   linear_ops[i]->take_step(-0.5+relaxation, steps[i]);
  //   // if (should_normalize)
  //   // linear_ops[i]->normalize();
  //   linear_ops[i]->get_inner_products(inner_products_x[i], inner_products_b[i]);
  // }
  // */
  // // double scale = 1;
  // // int si = -1;
  // // for (int i = 0; i < linear_ops.size(); ++i) {
  // //   double norm = linear_ops[i]->normalize();
  // //   if (norm != 0)
  // //     scale *= norm;
  // //   else
  // //     si = i;
  // // }
  // // linear_ops[si]->scale(scale);
  return std::sqrt(res);
}

double NonlinearGS::get_residual() const {
  std::vector<InnerProducts> coefficients_x(inner_products_x[0].size(),
                                            inner_products_x[0].back());
  std::vector<double> coefficients_b(inner_products_b[0].size());
  double r = 0;
  for (int i = 0; i < linear_ops.size(); ++i) {
    set_coefficients(i, coefficients_x, coefficients_b);
    double ri = linear_ops[i]->get_residual(coefficients_x, coefficients_b);
    r += std::pow(ri, 2);
  }
  return std::sqrt(r);
}

void NonlinearGS::set_coefficients(
      int i, std::vector<InnerProducts> &coefficients_x, 
      std::vector<double> &coefficients_b) const {
  for (int m = 0; m < coefficients_x.size(); ++m)
    coefficients_x[m] = 1;
  for (int n = 0; n < coefficients_b.size(); ++n)
    coefficients_b[n] = 1;
  for (int j = 0; j < linear_ops.size(); ++j) {
    if (i == j)
      continue;
    for (int m = 0; m < coefficients_x.size(); ++m)
      coefficients_x[m] *= inner_products_x[j][m];
    for (int n = 0; n < coefficients_b.size(); ++n)
      coefficients_b[n] *= inner_products_b[j][n];
  }
}

void NonlinearGS::enrich() {
  double last = std::nan("a");
  for (int i = 0; i < linear_ops.size(); ++i) {
    inner_products_x[i].push_back(inner_products_one);
    last = linear_ops[i]->enrich(last);
    linear_ops[i]->normalize();
    if (i > 0) {
      linear_ops[i]->get_inner_products(inner_products_x[i], 
                                        inner_products_b[i]);
    }
  }
}

void NonlinearGS::set_inner_products() {
  for (int i = 0; i < linear_ops.size(); ++i)
    linear_ops[i]->get_inner_products(inner_products_x[i], inner_products_b[i]);
}

void NonlinearGS::finalize() {
  linear_ops.back()->get_inner_products(inner_products_x.back(), 
                                        inner_products_b.back());
  const int num_modes = inner_products_x[0].size();
  // std::cout << "num_modes " << num_modes << std::endl;
  for (int i = 0; i < linear_ops.size(); ++i) {
    inner_products_all_x[i].push_back(inner_products_x[i]);
    inner_products_all_b[i].push_back(inner_products_b[i]);
    AssertDimension(inner_products_all_x[i].size(), num_modes);
    AssertDimension(inner_products_all_b[i].size(), num_modes);
    for (int m_row = 0; m_row < num_modes - 1; ++m_row) {
      inner_products_all_x[i][m_row].push_back(inner_products_one);
      // std::cout << inner_products_all_x[i][m_row].size() << std::endl;
      AssertDimension(inner_products_all_x[i][m_row].size(), num_modes);
      linear_ops[i]->get_inner_products(
          inner_products_all_x[i][m_row], inner_products_all_b[i][m_row],
          m_row, num_modes-1);
    }
  }
}

void NonlinearGS::update() {
  const int num_modes = inner_products_x[0].size();
  const int num_sources = inner_products_b[0].size();
  for (int i = 0; i < linear_ops.size(); ++i) {
    auto updatable = dynamic_cast<LinearUpdatableInterface*>(linear_ops[i]);
    if (updatable == NULL)
      continue;
    std::vector<std::vector<InnerProducts>> coefficients_x(
        num_modes, std::vector<InnerProducts>(num_modes, inner_products_one));
    std::vector<std::vector<double>> coefficients_b(
        num_modes, std::vector<double>(num_sources, 1.0));
    for (int j = 0; j < linear_ops.size(); ++j) {
      if (i == j)
        continue;
      for (int m_row = 0; m_row < num_modes; ++m_row) {
        for (int m_col = 0; m_col < num_modes; ++m_col)
          coefficients_x[m_row][m_col] *= inner_products_all_x[j][m_row][m_col];
        for (int s = 0; s < num_sources; ++s)
          coefficients_b[m_row][s] *= inner_products_all_b[j][m_row][s];
      }
    }
    updatable->update(coefficients_x, coefficients_b);
  }
}

void NonlinearGS::reweight() {
  
}

double NonlinearGS::line_search(
      const std::vector<dealii::Vector<double>> &steps) {
  // TODO: generalize this
  EnergyMgFull &energy_mg =
      dynamic_cast<EnergyMgFull&>(*linear_ops[0]);
  FixedSourceP<2, 2> &fixed_source_p =
      dynamic_cast<FixedSourceP<2, 2>&>(*linear_ops[1]);
  // assemble source
  const int num_groups = energy_mg.sources[0].size();
  const int num_qdofs = fixed_source_p.sources[0].size();
  const int num_sources = energy_mg.sources.size();
  dealii::BlockVector<double> v0(num_groups, num_qdofs);
  dealii::BlockVector<double> v1(v0.get_block_indices());
  dealii::BlockVector<double> v2(v0.get_block_indices());
  dealii::Vector<double> block(num_qdofs);
  for (int s = 0; s < num_sources; ++s) {
    for (int g = 0; g < num_groups; ++g) {
      block =  fixed_source_p.sources[s];
      v0.block(g).add(energy_mg.sources[s][g], block);
    }
  }
  v0 *= -1;
  double f0_norm = v0.l2_norm();
  const int num_modes = energy_mg.modes.size();
  for (int m = 0; m < num_modes; ++m) {
    expand_mode(v0, fixed_source_p.caches[m], energy_mg.modes[m]);
  }
  // higher-order terms
  const Transport<2, 2> &transport =
      dynamic_cast<const Transport<2, 2>&>(
          fixed_source_p.fixed_source.within_groups[0].transport.transport);
  const int num_ords = transport.quadrature.size(); //fixed_source_p.sources[0].n_blocks();
  const int num_dofs = transport.dof_handler.n_dofs(); //fixed_source_p.sources[0].block(0).size();
  Cache step_cache(1, num_ords, 1, num_dofs);
  step_cache.mode = steps[1];
  fixed_source_p.set_cache(step_cache);
  expand_mode(v1, step_cache, energy_mg.modes.back());
  expand_mode(v1, fixed_source_p.caches.back(), steps[0]);
  expand_mode(v2, step_cache, steps[0]);
  // get quartic coefficients
  AssertThrow(std::isfinite(f0_norm), dealii::ExcNumberNotFinite(f0_norm));
  std::vector<double> c(5);
  auto ip = [energy_mg, transport](const dealii::BlockVector<double> &a,
                                   const dealii::BlockVector<double> &b) {
    double c = 0;
    const int num_groups =  energy_mg.mgxs.total.size();
    for (int g = 0; g < num_groups; ++g) {
      int gg = num_groups - 1 - g;
      double width = std::log(energy_mg.mgxs.group_structure[gg+1]
                              /energy_mg.mgxs.group_structure[gg]);
      width = 1;
      c += std::pow(transport.inner_product(a, b), 2) / width;
    }
    // return c;
    return a * b;
  };
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < steps[i].size(); ++j)
      AssertThrow(std::isfinite(steps[i][j]), 
                  dealii::ExcNumberNotFinite(steps[i][j]));
  for (auto &v : {v0, v1, v2})
    for (int i = 0; i < v.size(); ++i)
      AssertThrow(std::isfinite(v[i]), dealii::ExcNumberNotFinite(v[i]));
  c[0] = ip(v0, v0); //v0 * v0;
  c[1] = 2 * ip(v1, v0);
  c[2] = ip(v1, v1) + 2 * ip(v2, v0);
  c[3] = 2 * ip(v1, v2);
  c[4] = ip(v2, v2);
  auto res = [](const std::vector<double> &c, double x) {
    double r = 0;
    for (int i = 0; i < c.size(); ++i) {
      // if (!std::isfinite(c[i]))
      //   continue;
      r += c[i] * std::pow(x, i);
    }
    return std::sqrt(std::abs(r));
  };
  std::cout << std::sqrt(c[0]) / f0_norm << " " << f0_norm <<std::endl;
  std::cout << res(c, 0) / f0_norm << " " << res(c, 1) / f0_norm << std::endl;
  // AssertThrow(std::isfinite(c[0]), dealii::ExcNumberNotFinite(c[0]));
  // AssertThrow(std::isfinite(c[1]), dealii::ExcNumberNotFinite(c[1]));
  // AssertThrow(std::isfinite(c[2]), dealii::ExcNumberNotFinite(c[2]));
  // AssertThrow(std::isfinite(c[3]), dealii::ExcNumberNotFinite(c[3]));
  // AssertThrow(std::isfinite(c[4]), dealii::ExcNumberNotFinite(c[4]));
  for (int i = 0; i < c.size(); ++i) {
    AssertThrow(std::isfinite(c[i]), dealii::ExcNumberNotFinite(c[i]));
  }
  // take derivative
  std::vector<double> dc(c.size()-1);
  for (int i = 0; i < dc.size(); ++i)
    dc[i] = (i+1) * c[i+1];
  // root the cubic
  double delta0 = std::pow(dc[2], 2) - 3 * dc[3] * dc[1];
  double delta1 = 2 * std::pow(dc[2], 3) 
                  - 9 * dc[3] * dc[2] * dc[1] 
                  + 27 * std::pow(dc[3], 2) * dc[0];
  double sqrt = std::sqrt(std::abs(std::pow(delta1, 2) - 4*std::pow(delta0, 3)));
  AssertThrow(std::isfinite(sqrt), dealii::ExcNumberNotFinite(sqrt));
  double cbrt = std::cbrt((delta1+sqrt)/2);
  AssertThrow(std::isfinite(sqrt), dealii::ExcNumberNotFinite(cbrt));
  if (std::abs(cbrt) < 1e-16)
    cbrt = std::cbrt((delta1-sqrt)/2);
  std::complex<double> u0(1, 0);
  std::complex<double> u1(-1./2, +std::cbrt(3)/2);
  std::complex<double> u2 = u1 * u1; //u2(-1./2, -std::cbrt(3)/2);
  std::vector<std::complex<double>> u = {u0, u1, u2};
  std::vector<std::complex<double>> roots(3);
  for (int i = 0; i < u.size(); ++i) {
    roots[i] = -(1./(3*dc[3])) * (dc[2] + u[i]*cbrt + delta0/(u[i]*cbrt));
    std::cout << roots[i] << " ";
  }
  return roots[0].real();
}

void NonlinearGS::expand_mode(dealii::BlockVector<double> &mode,
                              const pgd::sn::Cache &cache, 
                              const dealii::Vector<double> &mode_energy) {
  // TODO: generalize this
  EnergyMgFull &energy_mg =
      dynamic_cast<EnergyMgFull&>(*linear_ops[0]);
  FixedSourceP<2, 2> &fixed_source_p =
      dynamic_cast<FixedSourceP<2, 2>&>(*linear_ops[1]);
  // 
  const Transport<2, 2> &transport =
      dynamic_cast<const Transport<2, 2>&>(
          fixed_source_p.fixed_source.within_groups[0].transport.transport);
  const aether::sn::MomentToDiscrete<2, 2> &m2d =
      dynamic_cast<const aether::sn::MomentToDiscrete<2, 2>&>(
          fixed_source_p.fixed_source.m2d);
  const dealii::Quadrature<2> &quadrature = transport.quadrature;
  const dealii::DoFHandler<2> &dof_handler = transport.dof_handler;
  const int num_groups = energy_mg.mgxs.total.size();
  std::vector<dealii::types::global_dof_index> dof_indices(
      dof_handler.get_fe().dofs_per_cell);
  dealii::Vector<double> streamed_k(dof_indices.size());
  dealii::Vector<double> mass_inv_streamed_k(dof_indices.size());
  dealii::Vector<double> scattered(mode.block(0).size());
  AssertThrow(mode.block(0).size() == quadrature.size() * dof_handler.n_dofs(),
              dealii::ExcMessage("Wrong size"));
  m2d.vmult(scattered, cache.moments.block(0));
  for (int g = 0; g < num_groups; ++g) {
    int c = 0;
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
          ++cell, ++c) {
      if (!cell->is_locally_owned()) {
        --c;
        continue;
      }
      cell->get_dof_indices(dof_indices);
      const int j = cell->material_id();
      for (int n = 0; n < quadrature.size(); ++n) {
        for (int i = 0; i < dof_indices.size(); ++i) {
          const dealii::types::global_dof_index ni =
              n * dof_handler.n_dofs() + dof_indices[i];
          streamed_k[i] = cache.streamed.block(0)[ni];
        }
        dealii::FullMatrix<double> mass = transport.cell_matrices[c].mass;
        mass.gauss_jordan();
        mass.vmult(mass_inv_streamed_k, streamed_k);
        for (int i = 0; i < dof_indices.size(); ++i) {
          const dealii::types::global_dof_index ni =
              n * dof_handler.n_dofs() + dof_indices[i];
          double dof_m = 
              mass_inv_streamed_k[i] * mode_energy[g];
          dof_m += energy_mg.mgxs.total[g][j] * cache.mode.block(0)[ni] 
                    * mode_energy[g];
          for (int gp = 0; gp < num_groups; ++gp)
            dof_m += energy_mg.mgxs.scatter[g][gp][j] * scattered[ni]
                      * mode_energy[gp];
          AssertThrow(std::isfinite(mass_inv_streamed_k[i]),
                      dealii::ExcNumberNotFinite(mass_inv_streamed_k[i]));
          AssertThrow(std::isfinite(scattered[ni]),
                      dealii::ExcNumberNotFinite(scattered[ni]));
          AssertThrow(std::isfinite(mode_energy[g]), 
                      dealii::ExcNumberNotFinite(mode_energy[g]));
          AssertThrow(std::isfinite(dof_m), dealii::ExcNumberNotFinite(dof_m));
          mode.block(g)[ni] += dof_m;
          // std::cout << g << " " << n << " " << i << std::endl;
        }
      }
    }
  }
  return;
}

}  // namespace aether::pgd::sn