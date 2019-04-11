#pragma once

#include <array>
#include <iostream>

#include "dataset.hpp"

namespace perceptron {

template <int N>
class Perceptron {
public:
   int call(const Point<N>& x) const {
      return step(w0 + std::inner_product(x.cbegin(), x.cend(), weights.cbegin(), 0.0));
   }

   void train(const Dataset<N>& point)
   {
      const auto sgn = point.y - call(point.x);

      for (std::size_t i = 0; i < weights.size(); i++) {
         weights[i] += sgn*point.x[i];
         w0 += sgn;
      }
   }

   void print(std::ostream& ostr) const {
      ostr << '{' << w0;
      for (const auto& w: weights)
         ostr << ',' << w;
      ostr << '}';
   }

   void print_gnuplot_function(std::ostream& ostr) const {
      const auto wmax = weights.back();
      const int nx = static_cast<int>(weights.size()) - 1;

      ostr << "f(";

      for (int i = 0; i < nx; i++) {
         ostr << "x" << i;
         if (i < nx - 1)
            ostr << ',';
      }

      ostr << ") = " << -w0/wmax;

      for (int i = 0; i < nx; i++) {
         ostr << " + " << -weights[i]/wmax << "*x" << i;
      }

      ostr << '\n';
   }

private:
   Point<N> weights{};
   double w0{0.0};

   int step(double x) const {
      return x < 0 ? 0 : 1;
   }
};

}
