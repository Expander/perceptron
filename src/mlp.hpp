#pragma once

#include "dataset.hpp"

namespace perceptron {

template <int N>
class MLP
{
public:
   int run(const Point<N>& x) const
   {
      return 0;
   }


   void train(const Dataset<N>& point)
   {
   }

};

template <int N>
std::ostream& operator<<(std::ostream& ostr, const MLP<N>& p)
{
   return ostr;
}

} // namespace perceptron
