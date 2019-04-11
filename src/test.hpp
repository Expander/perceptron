#pragma once

#include <iostream>
#include <vector>

#include "dataset.hpp"

namespace perceptron {

struct Test_output {
   double mean_diff{0.0};

   friend std::ostream& operator<<(std::ostream&, const Test_output&);
};

std::ostream& operator<<(std::ostream& ostr, const Test_output& t)
{
   ostr << "Test results\n"
           "============\n"
        << "mean abs diff = " << t.mean_diff << " (should be < 0.5)"
        << '\n';
   return ostr;
}

template <class F, int N>
Test_output test(F f, const std::vector<Dataset<N>>& dataset)
{
   double mean_diff = 0.;

   for (const auto& d: dataset) {
      const auto result = f.call(d.x);
      mean_diff += std::abs(d.y - result);
   }

   mean_diff /= dataset.size();

   Test_output test_output;
   test_output.mean_diff = mean_diff;

   return test_output;
}

}
