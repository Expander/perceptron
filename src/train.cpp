#include "dataset.hpp"
#include "perceptron.hpp"
#include "mlp.hpp"
#include "test.hpp"

#include <fstream>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

int main(int argc, char* argv[])
{
   using namespace perceptron;

   std::string training_sample_file;
   std::string testing_sample_file;

   try {
      po::options_description desc{"Options"};
      desc.add_options()
         ("help,h", "This help message.")
         ("training-sample,t", po::value<std::string>(&training_sample_file)
          ->default_value("training_sample.txt"), "Training sample file.")
         ("testing-sample,e", po::value<std::string>(&testing_sample_file)
          ->default_value("testing_sample.txt"), "Testing sample file.");

      po::variables_map vm;
      po::store(po::parse_command_line(argc, argv, desc),vm);

      if (vm.count("help")) {
         std::cout << desc << std::endl;
         return EXIT_SUCCESS;
      }

      po::notify(vm);
   } catch (const po::error &ex) {
      std::cerr << "Error: " << ex.what() << '\n';
      return EXIT_FAILURE;
   }

   constexpr int N = 2;

   std::cout << "Reading training sample from \"" << training_sample_file << "\"";
   const auto training_sample = read_from_file<N>(training_sample_file);
   std::cout << " (" << training_sample.size() << " data points)\n";

   std::cout << "Reading testing sample from \"" << testing_sample_file << "\"";
   const auto testing_sample = read_from_file<N>(testing_sample_file);
   std::cout << " (" << testing_sample.size() << " data points)\n";

   std::cout << "training ..." << std::endl;

   Perceptron<N> slp;
   MLP<N,N> mlp;

   slp.train(training_sample);
   mlp.train(training_sample);

   std::cout << "\ntesting SLP ...\n\n";
   const auto to_slp = test(slp, testing_sample);
   std::cout << to_slp;

   std::cout << "\ntesting MLP ...\n\n";
   const auto to_mlp = test(mlp, testing_sample);
   std::cout << to_mlp;

   std::cout << "\nPerceptron layout: " << slp << std::endl;

   std::cout << "\nMLP layout: " << mlp << std::endl;

   std::ofstream slp_script("perceptron.gnuplot");
   print_gnuplot_function(slp, slp_script);
   slp_script << "plot 'training_sample.txt' u 1:2:3 w points palette, f(x)"
              << std::endl;

   return 0;
}
