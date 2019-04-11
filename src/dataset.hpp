#pragma once

#include <array>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace perceptron {

namespace {

double make_random(double min_, double max_)
{
   static std::random_device rd;
   static std::mt19937 rng(rd());
   static std::uniform_real_distribution<double> uni(min_, max_);

   return uni(rng);
}

} // anonymous namespace

template <int N>
using Point = std::array<double,N>;

template <int N>
struct Dataset {
   Point<N> x; // input vector
   double y; // output
};

template <int N, class Func>
auto make_dataset(Func f, int npoints) -> std::vector<Dataset<N>>
{
   std::vector<Dataset<N>> dataset(npoints);

   for (int i = 0; i < npoints; i++) {
      for (int n = 0; n < N; n++)
         dataset[i].x[n] = make_random(0., 1.);
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

}
