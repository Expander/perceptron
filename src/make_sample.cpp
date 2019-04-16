#include "dataset.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

int main(int argc, char* argv[])
{
   using namespace perceptron;

   long npoints = 0;
   std::string output_file;

   try {
      po::options_description desc{"Options"};
      desc.add_options()
         ("help,h", "This help message.")
         ("number,n", po::value<long>(&npoints)
          ->default_value(10000), "Number of points.")
         ("output,o", po::value<std::string>(&output_file)
          ->default_value(""), "Output file.");

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

   // and
   auto f = [](const Point<N>& x) -> int {
      return x[0] > 0.5 && x[1] > 0.5 ? 1 : 0;
   };

   std::cout << "generating " << npoints << " data points ..." << std::endl;

   const auto sample = make_dataset<N>(f, npoints);

   if (output_file.empty()) {
      write_to_stream(std::cout, sample);
   } else {
      std::cout << "saving data points in " << output_file << '\n';
      write_to_file(output_file, sample);
   }

   return 0;
}
