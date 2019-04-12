#include "dataset.hpp"
#include "perceptron.hpp"
#include "mlp.hpp"
#include "test.hpp"

int main(int argc, char* argv[])
{
   using namespace perceptron;

   long npoints = 10000;

   if (argc > 1)
      npoints = atol(argv[1]);

   constexpr int N = 2;

   // and
   auto f = [](const Point<N>& x) -> int {
      return x[0] > 0.5 && x[1] > 0.5 ? 1 : 0;
   };

   std::cout << "generating data (" << npoints << " points) ..." << std::endl;

   const auto training_sample = make_dataset<N>(f, npoints);
   write_to_file("training_sample.txt", training_sample);

   const auto testing_sample = make_dataset<N>(f, npoints);
   write_to_file("testing_sample.txt", testing_sample);

   std::cout << "training ..." << std::endl;

   Perceptron<N> slp;
   MLP<N,N> mlp;

   for (const auto& d : training_sample)
      slp.train(d);

   for (const auto& d : training_sample)
      mlp.train(d);

   std::cout << "\ntesting SLP ...\n\n";
   const auto to_slp = test(slp, testing_sample);
   std::cout << to_slp;

   std::cout << "\ntesting MLP ...\n\n";
   const auto to_mlp = test(mlp, testing_sample);
   std::cout << to_mlp;

   std::cout << "\nPerceptron layout: " << slp << std::endl;

   std::cout << "\nMLP layout: " << mlp << std::endl;

   std::cout << '\n';
   std::cout << "Gnuplot script:\n";
   print_gnuplot_function(slp, std::cout);
   std::cout << "plot 'training_sample.txt' u 2:3:1 w points palette, f(x)"
             << std::endl;

   return 0;
}
