#pragma once

#include "dataset.hpp"

#include <array>

namespace perceptron {

namespace detail {

template <int N>
struct Hidden_layer
{
   std::array<Point<N>,N> w;
   Point<N> w0;
};

template <int N>
struct Output_layer
{
   Point<N> w;
   double w0;
};

} // namespace detail

template <int N, int L>
class MLP
{
public:
   double run(const Point<N>& x) const
   {
      Point<N> a = x;

      for (const auto& l : hidden_layers)
         update(a, l);

      return combine(a, output_layer);
   }

   void train(const std::vector<Dataset<N>>& dataset)
   {
   }

private:
   std::array<detail::Hidden_layer<N>,L> hidden_layers;
   detail::Output_layer<N> output_layer;

   static double flog(double z) { return 1. / (1. + std::exp(-z)); }

   static void update(Point<N>& a, const detail::Hidden_layer<N>& l)
   {
      Point<N> net{};

      for (std::size_t i = 0; i < N; i++) {
         const auto mp =
            std::inner_product(l.w[i].cbegin(), l.w[i].cend(), a.cbegin(), 0.0);
         net[i] = l.w0[i] + mp;
      }

      for (std::size_t i = 0; i < N; i++) {
         a[i] = flog(net[i]);
      }
   }

   static double combine(Point<N>& a, const detail::Output_layer<N>& l)
   {
      double net = 0.0;

      for (std::size_t i = 0; i < N; i++) {
         const auto mp =
            std::inner_product(l.w.cbegin(), l.w.cend(), a.cbegin(), 0.0);
         net = l.w0 + mp;
      }

      return flog(net);
   }
};

template <int N, int L>
std::ostream& operator<<(std::ostream& ostr, const MLP<N, L>& p)
{
   return ostr;
}

} // namespace perceptron
