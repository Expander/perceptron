#pragma once

#include <array>
#include <fstream>
#include <random>
#include <sstream>
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

template <int N>
struct Reader {
   template <typename Iterator>
   static void read_from(std::istream& istr, Iterator it)
   {
      istr >> *it;
      Reader<N-1>::read_from(istr, ++it);
   }
};

template <>
struct Reader<1> {
   template <typename Iterator>
   static void read_from(std::istream& istr, Iterator it)
   {
      istr >> *it;
      ++it;
   }
};

template <>
struct Reader<0> {
   template <typename Iterator>
   static void read_from(std::istream& istr, Iterator it)
   {
   }
};

} // namespace detail

template <int N>
using Point = std::array<double, N>;

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
std::vector<Dataset<N>> read_from_file(const std::string& filename)
{
   std::ifstream fst(filename);
   std::vector<Dataset<N>> vec;

   std::string line;
   while (std::getline(fst, line)) {
      std::istringstream iss(line);
      Dataset<N> dataset;

      detail::Reader<N>::read_from(iss, dataset.x.begin());
      iss >> dataset.y;

      vec.push_back(std::move(dataset));
   }

   return vec;
}

template <int N>
void write_to_stream(std::ostream& ostr,
                     const std::vector<Dataset<N>>& dataset)
{
   for (const auto& d : dataset) {
      for (const auto& x : d.x)
         ostr << x << '\t';
      ostr << d.y << '\n';
   }
}

template <int N>
void write_to_file(const std::string& filename,
                   const std::vector<Dataset<N>>& dataset)
{
   std::ofstream fst(filename);
   write_to_stream(fst, dataset);
}

template <std::size_t N>
std::ostream& operator<<(std::ostream& ostr, const Point<N>& p)
{
   ostr << '[';
   for (std::size_t i = 0; i < p.size(); i++) {
      ostr << p[i];
      if (i + 1 < p.size())
         ostr << ", ";
   }
   ostr << ']';

   return ostr;
}

} // namespace perceptron
