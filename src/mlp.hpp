#pragma once

#include "dataset.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace perceptron {

namespace detail {

template <int N>
struct Hidden_layer
{
   std::array<Point<N>,N> w; ///< w[j][i] = weight for input neuron i and output j
   Point<N> w0;
};

template <int N>
struct Output_layer
{
   Point<N> w;
   double w0;
};

double activation(double z) { return 1. / (1. + std::exp(-z)); }

/// appends array to vector
template <std::size_t N>
void append(std::vector<double>& vec, const std::array<double, N>& a)
{
   vec.insert(vec.end(), a.cbegin(), a.cend());
}

/// appends array to vector
void append(std::vector<double>& vec, double a) { vec.push_back(a); }

/// exctracts array from vector
template <std::size_t N>
void extract(const std::vector<double>& vec, std::array<double, N>& a,
             std::size_t start)
{
   if (start + N - 1 > vec.size())
      throw std::runtime_error("Index out of bounds!");

   std::copy_n(vec.cbegin() + start, N, a.begin());
}

/// exctracts array from vector
void extract(const std::vector<double>& vec, double& a,
             std::size_t start)
{
   if (start > vec.size())
      throw std::runtime_error("Index out of bounds!");

   a = *(vec.cbegin() + start);
}

template <std::size_t N>
double scp(const Point<N>& a, const Point<N>& b)
{
   return std::inner_product(a.cbegin(), a.cend(), b.cbegin(), 0.0);
}

template <typename T>
T sqr (T x) { return x*x; }

} // namespace detail

template <int N, int L>
class MLP
{
public:
   double run(const Point<N>& x) const
   {
      Point<N> a = x;

      for (const auto& l : hidden_layers) {
         run_hidden_layer(a, l);
      }

      return run_output_layer(a);
   }

   void train(const std::vector<Dataset<N>>& dataset, std::size_t epochs = 100, double learning_rate = 1.0)
   {
      const double max_grad = 1e-10;
      std::size_t epoch = 0;
      auto weights = get_weights();
      auto gr = grad(weights, dataset);

      while (norm(gr) > max_grad && epoch++ < epochs) {
         for (std::size_t i = 0; i < weights.size(); i++) {
            weights[i] -= learning_rate*gr[i];
         }

         gr = grad(weights, dataset);
      };

      set_weights(weights);
   }

   template <int NX, int LX>
   friend std::ostream& operator<<(std::ostream&, const MLP<NX, LX>&);

private:
   std::array<detail::Hidden_layer<N>,L> hidden_layers;
   detail::Output_layer<N> output_layer;

   static void run_hidden_layer(Point<N>& a, const detail::Hidden_layer<N>& l)
   {
      Point<N> net{};

      for (std::size_t i = 0; i < N; i++) {
         net[i] = l.w0[i] + detail::scp(l.w[i], a);
      }

      for (std::size_t i = 0; i < N; i++) {
         a[i] = detail::activation(net[i]);
      }
   }

   double run_output_layer(Point<N>& a) const
   {
      const auto mp = detail::scp(output_layer.w, a);
      const auto net = output_layer.w0 + mp;

      return detail::activation(net);
   }

   /// calculate gradient of err(w,D)
   std::vector<double> grad(std::vector<double>& w,
                            const std::vector<Dataset<N>>& dataset) const
   {
      const double eps = 1e-10;

      std::vector<double> gr;
      gr.reserve(w.size());

      for (std::size_t i = 0; i < w.size(); i++) {
         auto we = w;
         we[i] += eps;
         gr.push_back((err(we, dataset) - err(w, dataset)) / eps);
      }

      return gr;
   }

   /// calculate error of given MLP
   static double err(const MLP<N, L>& mlp, const std::vector<Dataset<N>>& dataset)
   {
      double e = 0.;

      for (const auto& d : dataset)
         e += loss(mlp, d);

      return e/dataset.size();
   }

   /// loss for a single point
   static double loss(const MLP<N, L>& mlp, const Dataset<N>& d)
   {
      return 0.5 * detail::sqr(mlp.run(d.x) - d.y);
   }

   /// calculate error
   double err(const std::vector<Dataset<N>>& dataset) const
   {
      return err(*this);
   }

   /// calculate error for given set of weights
   double err(const std::vector<double>& w,
              const std::vector<Dataset<N>>& dataset) const
   {
      auto mlp(*this);
      mlp.set_weights(w);
      return err(mlp, dataset);
   }

   /// returns all weights
   std::vector<double> get_weights() const
   {
      std::vector<double> vec;

      for (const auto& l : hidden_layers) {
         detail::append(vec, l.w0);
         for (const auto& wa : l.w) {
            detail::append(vec, wa);
         }
      }

      detail::append(vec, output_layer.w0);
      detail::append(vec, output_layer.w);

      return vec;
   }

   void set_weights(const std::vector<double>& vec)
   {
      std::size_t offset = 0;

      for (auto& l : hidden_layers) {
         detail::extract(vec, l.w0, offset);
         offset += N;
         for (auto& wa : l.w) {
            detail::extract(vec, wa, offset);
            offset += N;
         }
      }

      detail::extract(vec, output_layer.w0, offset++);
      detail::extract(vec, output_layer.w, offset);
      offset += N;
   }

   /// norm of vector
   static double norm(std::vector<double>& vec)
   {
      double n = 0.;
      for (const auto v : vec)
         n += detail::sqr(v);
      return std::sqrt(n);
   }
};

template <int N, int L>
std::ostream& operator<<(std::ostream& ostr, const MLP<N, L>& mlp)
{
   ostr << "{\n";

   for (const auto& l : mlp.hidden_layers) {
      ostr << "[" << l.w0;
      for (const auto& w : l.w) {
         ostr << ",\n " << w;
      }
      ostr << "],\n";
   }

   ostr << '[' << mlp.output_layer.w0 << ",\n "
        << mlp.output_layer.w << ']';

   ostr << "\n}";

   return ostr;
}

} // namespace perceptron
