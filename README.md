Perceptron
==========

Ad-hoc implementation of perceptrons.

Building
--------

    meson build
    cd build
    ninja

Running
-------

Generating training and testing samples:

    ./make_sample.x -o training_sample.txt
    ./make_sample.x -o testing_sample.txt

Training the classifier(s):

    ./train.x -t training_sample.txt -e testing_sample.txt

The results can be plotted like this:

    gnuplot -p perceptron.gnuplot
