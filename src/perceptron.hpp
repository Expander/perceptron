#pragma once

#include <array>
#include <iostream>
#include <numeric>

#include "dataset.hpp"

namespace perceptron {

template <int N>
class Perceptron
{
public:
   const Point<N>& get_weights() const { return weights; }
   double get_bias_weight() const { return w0; }

   int run(const Point<N>& x) const
   {
      return step(
         w0 + std::inner_product(x.cbegin(), x.cend(), weights.cbegin(), 0.0));
   }

   void train(const std::vector<Dataset<N>>& dataset)
   {
      for (const auto& d : dataset)
         train(d);
   }

private:
   Point<N> weights{};
   double w0{0.0};

   int step(double x) const { return x < 0 ? 0 : 1; }

   void train(const Dataset<N>& point)
   {
      const auto sgn = point.y - run(point.x);

      w0 += sgn;

      for (std::size_t i = 0; i < weights.size(); i++)
         weights[i] += sgn * point.x[i];
   }
};

template <int N>
void print_gnuplot_function(const Perceptron<N>& p, std::ostream& ostr)
{
   const auto w0 = p.get_bias_weight();
   const auto& weights = p.get_weights();
   const auto wmax = weights.back();
   const int nx = static_cast<int>(weights.size()) - 1;

   ostr << "f(";

   for (int i = 0; i < nx; i++) {
      ostr << 'x' << i;
      if (i < nx - 1)
         ostr << ',';
   }

   ostr << ") = " << -w0 / wmax;

   for (int i = 0; i < nx; i++) {
      ostr << " + " << -weights[i] / wmax << "*x" << i;
   }

   ostr << '\n';
}

template <int N>
std::ostream& operator<<(std::ostream& ostr, const Perceptron<N>& p)
{
   const auto w0 = p.get_bias_weight();
   const auto& weights = p.get_weights();

   ostr << '{' << w0;
   for (const auto& w : weights)
      ostr << ", " << w;
   ostr << '}';

   return ostr;
}

} // namespace perceptron
