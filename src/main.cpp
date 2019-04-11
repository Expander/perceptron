#include "dataset.hpp"
#include "perceptron.hpp"
#include "test.hpp"

int main(int argc, char* argv[])
{
   using namespace perceptron;

   int npoints = 10000;

   if (argc > 1)
      npoints = atoi(argv[1]);

   constexpr int N = 2;
   Perceptron<N> perceptron;

   // and
   auto f = [] (const Point<N>& x) -> int {
      return x[0] > 0.5 && x[1] > 0.5;
   };

   std::cout << "generating data (" << npoints << " points) ..." << std::endl;

   const auto training_sample = make_dataset<N>(f, npoints);
   write_to_file("training_sample.txt", training_sample);

   const auto testing_sample = make_dataset<N>(f, npoints);
   write_to_file("testing_sample.txt", testing_sample);

   std::cout << "training ..." << std::endl;

   for (const auto& d: training_sample)
      perceptron.train(d);

   std::cout << "testing ...\n\n";
   const auto test_output = test(perceptron, testing_sample);
   std::cout << test_output;

   std::cout << "\nPerceptron layout: ";
   perceptron.print(std::cout);
   std::cout << std::endl;

   std::cout << '\n';
   std::cout << "Gnuplot script:\n";
   perceptron.print_gnuplot_function(std::cout);
   std::cout << "plot 'training_sample.txt' u 2:3:1 w points palette, f(x)" << std::endl;

   return 0;
}
