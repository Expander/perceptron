project('perceptron', 'cpp',
        default_options : ['cpp_std=c++14', 'buildtype=release'])

conf_inc = include_directories('src')

boost_dep = dependency('boost', modules : ['program_options'])

main_src = ['src/main.cpp']

executable('run.x', main_src,
           include_directories : conf_inc,
           dependencies: boost_dep)
