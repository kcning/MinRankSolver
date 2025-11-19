NAME
    mrsolver

DESCRIPTION
    C implementation of the Multi-homogeneous XL algorithm for solving MinkRank
    problems. Please note that the current program assume GF(16) and has to be
    expanded to support arbitrary fields.

DOCUMENTATION
    Please refer to our paper: https://eprint.iacr.org/2025/2060

DEPENDENCIES
    cmake, make, gcc

OPTIONAL DEPENDENCIES
    none

OPTIONS
    Please see the help message (--help)

BUILD
    run:
        $ mkdir build && cd build && cmake .. && make -j mrsolver

    An executable 'mrsolver' should appear in dir 'build/src'
