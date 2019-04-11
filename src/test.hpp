#pragma once

#include <iostream>
#include <vector>

#include "dataset.hpp"

namespace perceptron {

struct Test_output {
   std::size_t sample_size{0};
   std::size_t positive{0};
   std::size_t negative{0};

   std::size_t true_positive{0};
   std::size_t true_negative{0};
   std::size_t false_positive{0};
   std::size_t false_negative{0};

   double true_positive_rate{0};
   double true_negative_rate{0};
   double false_positive_rate{0};
   double false_negative_rate{0};

   double mean_diff{0.0};

   friend std::ostream& operator<<(std::ostream&, const Test_output&);
};

std::ostream& operator<<(std::ostream& ostr, const Test_output& t)
{
   ostr << "Test results\n"
           "============\n"
        << "sample size = " << t.sample_size << '\n'
        << "positive = " << t.positive << '\n'
        << "negative = " << t.negative << '\n'
        << "true positive = " << t.true_positive << '\n'
        << "true negative = " << t.true_negative << '\n'
        << "false positive = " << t.false_positive << '\n'
        << "false negative = " << t.false_negative << '\n'
        << "true positive rate = " << t.true_positive_rate << '\n'
        << "true negative rate = " << t.true_negative_rate << '\n'
        << "false positive rate = " << t.false_positive_rate << '\n'
        << "false negative rate = " << t.false_negative_rate << '\n'
        << "mean abs diff = " << t.mean_diff << " (should be < 0.5)"
        << '\n';
   return ostr;
}

template <class F, int N>
Test_output test(F f, const std::vector<Dataset<N>>& dataset)
{
   Test_output to;
   to.sample_size = dataset.size();

   for (const auto& d: dataset) {
      const auto y = f.call(d.x);

      if (d.y && y) to.true_positive++;
      if (!d.y && !y) to.true_negative++;
      if (d.y && !y) to.false_negative++;
      if (!d.y && y) to.false_positive++;

      to.mean_diff += std::abs(d.y - y);
   }

   to.positive = to.true_positive + to.false_negative;
   to.negative = to.true_negative + to.false_positive;

   to.true_positive_rate  = static_cast<double>(to.true_positive ) / to.positive;
   to.true_negative_rate  = static_cast<double>(to.true_negative ) / to.negative;
   to.false_positive_rate = static_cast<double>(to.false_positive) / to.negative;
   to.false_negative_rate = static_cast<double>(to.false_negative) / to.positive;

   to.mean_diff /= to.sample_size;

   return to;
}

}
