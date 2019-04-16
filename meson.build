project('perceptron', 'cpp',
        default_options : ['cpp_std=c++14', 'buildtype=release'])

conf_inc = include_directories('src')

boost_dep = dependency('boost', modules : ['program_options'])

train_src = ['src/train.cpp']

executable('train.x', train_src,
           include_directories : conf_inc,
           dependencies: boost_dep)

make_sample_src = ['src/make_sample.cpp']

executable('make_sample.x', make_sample_src,
           include_directories : conf_inc,
           dependencies: boost_dep)
