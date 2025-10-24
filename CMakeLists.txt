cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(mrs LANGUAGES C)

# set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message("Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# ========================================================================== #
# language standard requirements
# ========================================================================== #

if(NOT DEFINED CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED true)
endif()

# ========================================================================== #
# compilation flags
# ========================================================================== #

set(CMAKE_POSITION_INDEPENDENT_CODE OFF)

include(CheckPIESupported)
check_pie_supported()

include(CheckCCompilerFlag)

# dump compilation commands for clangd
# clangd should load the generated compile-commands.json automatically
# under directory 'build'
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

check_c_compiler_flag("-flto" COMPILER_C_LTO)
if(COMPILER_C_LTO)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -flto=auto"
        CACHE STRING "Enable link time optimization" FORCE
    )
endif()

check_c_compiler_flag("-Wpedantic" COMPILER_C_PEDANTIC)
if(COMPILER_C_PEDANTIC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wpedantic"
        CACHE STRING "Show warning messages demanded by strict ISO C" FORCE
    )
endif()

check_c_compiler_flag("-Wall" COMPILER_C_ALL_WARN)
if(COMPILER_C_ALL_WARN)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall"
        CACHE STRING "Show all warning messages" FORCE
    )
endif()

check_c_compiler_flag("-Wextra" COMPILER_C_EXTRA_WARN)
if(COMPILER_C_EXTRA_WARN)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wextra"
        CACHE STRING "Show extra warning messages" FORCE
    )
endif()

check_c_compiler_flag("-pipe" COMPILER_C_PIPE)
if(COMPILER_C_PIPE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe"
        CACHE STRING "Speed up compilation by piping" FORCE
    )
endif()

check_c_compiler_flag("-march=native" COMPILER_C_ARCH)
if(COMPILER_C_ARCH)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native"
        CACHE STRING "Optimize based on the architecture" FORCE
    )
endif()

check_c_compiler_flag("-mtune=native" COMPILER_C_TUNE)
if(COMPILER_C_TUNE)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -mtune=native"
        CACHE STRING "Fine-tune the program based on the CPU" FORCE
    )
endif()

# NOTE: do not turn on -Ofast or -ffast-math, which turn on
# -funsafe-math-optimizations. This unsafe optimization breaks Gauss-Jordan
# elimination implementation optimized with AVX floating-point domain instructions.
check_c_compiler_flag("-fallow-store-data-races" COMPILER_C_UNSAFE_DATA_STORE)
if(COMPILER_C_UNSAFE_DATA_STORE)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fallow-store-data-races"
        CACHE STRING "Enable unsafe data store optimization" FORCE
    )
endif()

check_c_compiler_flag("-fno-math-errno" COMPILER_C_NO_MATH_ERRNO)
if(COMPILER_C_NO_MATH_ERRNO)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fno-math-errno"
        CACHE STRING "Do not set errno in math functions" FORCE
    )
endif()

check_c_compiler_flag("-ffinite-math-only" COMPILER_C_FINITE_MATH)
if(COMPILER_C_FINITE_MATH)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -ffinite-math-only"
        CACHE STRING "Assume no NaN and +-Infs in floating-point arithmetics" FORCE
    )
endif()

option(USE_UINT24 "Use uint24 for Macaulay matrix" ON) # enabled by default
if(USE_UINT24)
    add_definitions(-DBLK_LANCZOS_UINT24_MAC)
endif(USE_UINT24)

option(BLK_COL_STATS "Collect statistics for analysis" OFF)
if(BLK_COL_STATS)
    add_definitions(-DBLK_LANCZOS_COLLECT_STATS)
endif(BLK_COL_STATS)

option(BLK_BLAKE_UNSAFE "Do not clear memory after Blake2s computation" ON)
if(BLK_BLAKE_UNSAFE)
    add_definitions(-DBLK_LANCZOS_BLAKE2S_NO_CLEANUP)
endif(BLK_BLAKE_UNSAFE)

option(GFA_IDX_LARGE "Use 56 bits for indices in Macaulay matrices" ON)
if(GFA_IDX_LARGE)
    add_definitions(-DGFA_IDX_SIZE_64)
endif(GFA_IDX_LARGE)

# ========================================================================== #
# paths
# ========================================================================== #

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

set(mrs_INSTALL_BIN_DIR ${PROJECT_SOURCE_DIR}/bin)
set(mrs_INSTALL_LIB_DIR ${PROJECT_SOURCE_DIR}/lib)
set(mrs_INSTALL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(mrs_HEADERS_DIR ${PROJECT_SOURCE_DIR}/src/mrs)

include_directories(${mrs_INSTALL_INCLUDE_DIR})
include_directories(${mrs_HEADERS_DIR})

# ========================================================================== #
# subdirectories
# ========================================================================== #

add_subdirectory(src)
