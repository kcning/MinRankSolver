cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(mrsolver LANGUAGES C)

add_subdirectory(mrs)
set(SRC main.c)

add_executable(mrsolver ${SRC})
set_target_properties(mrsolver PROPERTIES LINKER_TYPE DEFAULT)
set_property(TARGET mrsolver PROPERTY POSITION_INDEPENDENT_CODE FALSE)
target_link_libraries(mrsolver m mrs pthread)
install(TARGETS mrsolver DESTINATION ${mrs_INSTALL_BIN_DIR})
