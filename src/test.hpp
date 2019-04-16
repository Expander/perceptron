#pragma once

#include <cmath>
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

   double true_positive_rate{0.};
   double true_negative_rate{0.};
   double false_positive_rate{0.};
   double false_negative_rate{0.};

   double positive_predicive_value{0.};
   double negative_predicive_value{0.};
   double false_discovery_rate{0.};
   double false_omission_rate{0.};

   double accuracy{0.};
   double f1_score{0.};
   double matthews_correlation_coefficient{0.};
   double bookmaker_informednes{0.};
   double markednes{0.};

   double mean_diff{0.0};

   friend std::ostream& operator<<(std::ostream&, const Test_output&);
};

std::ostream& operator<<(std::ostream& ostr, const Test_output& t)
{
   ostr << "Test results\n"
           "============\n"
        << "sample size = " << t.sample_size << '\n'
        << "positive (P) = " << t.positive << '\n'
        << "negative (N) = " << t.negative << '\n'
        << "true positive (TP) = " << t.true_positive << '\n'
        << "true negative (TN) = " << t.true_negative << '\n'
        << "false positive (FP) = " << t.false_positive << '\n'
        << "false negative (FN) = " << t.false_negative << '\n'
        << "true positive rate (TPR) = " << t.true_positive_rate << '\n'
        << "true negative rate (TNR) = " << t.true_negative_rate << '\n'
        << "false positive rate (FPR) = " << t.false_positive_rate << '\n'
        << "false negative rate (FNR) = " << t.false_negative_rate << '\n'
        << "positive predicive value (PPV) = " << t.positive_predicive_value
        << '\n'
        << "negative predicive value (NPV) = " << t.negative_predicive_value
        << '\n'
        << "false discovery rate (FDR) = " << t.false_discovery_rate << '\n'
        << "false omission rate (FOR) = " << t.false_omission_rate << '\n'
        << "accuracy (ACC) = " << t.accuracy << '\n'
        << "F1 score = " << t.f1_score << '\n'
        << "matthews correlation coefficient (MCC) = "
        << t.matthews_correlation_coefficient << '\n'
        << "bookmaker informednes (BM) = " << t.bookmaker_informednes << '\n'
        << "markednes (MK) = " << t.markednes << '\n'
        << "Point in ROC space: (" << t.false_positive_rate << ", " << t.true_positive_rate << ")\n"
        << "mean abs diff = " << t.mean_diff << " (should be < 0.5)" << '\n';
   return ostr;
}

template <class Classifier, int N>
Test_output test(Classifier f, const std::vector<Dataset<N>>& dataset)
{
   Test_output to;
   to.sample_size = dataset.size();

   for (const auto& d : dataset) {
      const auto y = f.run(d.x);

      if (d.y && y)
         to.true_positive++;
      if (!d.y && !y)
         to.true_negative++;
      if (d.y && !y)
         to.false_negative++;
      if (!d.y && y)
         to.false_positive++;

      to.mean_diff += std::abs(d.y - y);
   }

   to.positive = to.true_positive + to.false_negative;
   to.negative = to.true_negative + to.false_positive;

   to.true_positive_rate = static_cast<double>(to.true_positive) / to.positive;
   to.true_negative_rate = static_cast<double>(to.true_negative) / to.negative;
   to.false_positive_rate =
      static_cast<double>(to.false_positive) / to.negative;
   to.false_negative_rate =
      static_cast<double>(to.false_negative) / to.positive;

   to.positive_predicive_value = static_cast<double>(to.true_positive) /
                                 (to.true_positive + to.false_positive);
   to.negative_predicive_value = static_cast<double>(to.true_negative) /
                                 (to.true_negative + to.false_negative);
   to.false_discovery_rate = static_cast<double>(to.false_positive) /
                             (to.false_positive + to.true_positive);
   to.false_omission_rate = static_cast<double>(to.false_negative) /
                            (to.false_negative + to.true_negative);

   to.accuracy =
      static_cast<double>(to.true_positive + to.true_negative) / to.sample_size;
   to.f1_score = 2.0 * to.positive_predicive_value * to.true_positive_rate /
                 (to.positive_predicive_value + to.true_positive_rate);
   to.matthews_correlation_coefficient =
      (to.true_positive * to.true_negative -
       to.false_positive * to.false_negative) /
      std::sqrt((to.true_positive + to.false_positive) *
                (to.true_positive + to.false_negative) *
                (to.true_negative + to.false_positive) *
                (to.true_negative + to.false_negative));
   to.bookmaker_informednes =
      to.true_positive_rate + to.true_negative_rate - 1.0;
   to.markednes =
      to.positive_predicive_value + to.negative_predicive_value - 1.0;

   to.mean_diff /= to.sample_size;

   return to;
}

} // namespace perceptron
