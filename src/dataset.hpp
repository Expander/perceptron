#pragma once

#include <array>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace perceptron {

namespace detail {

inline double make_random(double min_, double max_)
{
   static std::random_device rd;
   static std::mt19937 rng(rd());
   static std::uniform_real_distribution<double> uni(min_, max_);

   return uni(rng);
}

} // namespace detail

template <int N>
using Point = std::array<double,N>;

template <int N>
struct Dataset {
   Point<N> x; ///< N-dimensional data vector
   int y;      ///< class (0 = H0, 1 = H1)
};

template <int N, class Func>
auto make_dataset(Func f, long npoints) -> std::vector<Dataset<N>>
{
   std::vector<Dataset<N>> dataset(npoints);

   for (long i = 0; i < npoints; i++) {
      for (long n = 0; n < N; n++)
         dataset[i].x[n] = detail::make_random(0., 1.);
      dataset[i].y = f(dataset[i].x);
   }

   return dataset;
}

template <int N>
void write_to_file(const std::string& filename, const std::vector<Dataset<N>>& dataset)
{
   std::ofstream fst(filename);

   for (const auto& d: dataset) {
      fst << d.y;
      for (const auto& x: d.x)
         fst << "\t" << x;
      fst << '\n';
   }
}

} // namespace perceptron
