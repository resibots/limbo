cmake_minimum_required (VERSION 3.0.2)
project(limbo VERSION 2.1 DESCRIPTION "limbo LIbrary for Model-Based Optimization")

# Cpp standard
set(CMAKE_CXX_STANDARD 14)

# # Compiler options
# add_compile_options(-Wall -Wextra -Wpedantic)

# # Dependencies
# # Eigen
# find_package(Eigen3 REQUIRED)

# # Boost
# set(Boost_USE_STATIC_LIBS OFF)
# set(Boost_USE_MULTITHREADED ON)
# set(Boost_USE_STATIC_RUNTIME OFF)
# find_package(Boost 1.45.0 REQUIRED COMPONENTS system filesystem thread unit_test_framework)
# if(Boost_FOUND)
#   include_directories(${Boost_INCLUDE_DIRS})
# endif()

# # TBB
# find_package(TBB REQUIRED)

# # # NLOpt
# find_package(NLopt REQUIRED)
# if(NLopt_FOUND)
#   include_directories(${NLOPT_INCLUDE_DIRS})
#   add_definitions(-DUSE_NLOPT)
# endif()

# # Add library
# add_library(limbo INTERFACE IMPORTED GLOBAL) # Interface because it's header only

include(ExternalProject)
ExternalProject_Add(limbo_src
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  CONFIGURE_COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR} && ./waf configure
  BUILD_COMMAND     cd ${CMAKE_CURRENT_SOURCE_DIR} && ./waf build
  INSTALL_COMMAND ""
)

# Install
include(GNUInstallDirs)
# Install binaries
install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/build/src/liblimbo.a
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
# Install headers
install(DIRECTORY
          ${CMAKE_CURRENT_SOURCE_DIR}/src/limbo
          ${CMAKE_CURRENT_SOURCE_DIR}/src/external
          ${CMAKE_CURRENT_SOURCE_DIR}/src/hv
          ${CMAKE_CURRENT_SOURCE_DIR}/src/ehvi
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )

# Export
include(CMakePackageConfigHelpers)
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})

configure_package_config_file(cmake/limboConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/limboConfig.cmake
  INSTALL_DESTINATION ${LIB_INSTALL_DIR}/limbo/cmake
  PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/limboConfigVersion.cmake
  VERSION 0.0.0
  COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/limboConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/limboConfigVersion.cmake
        DESTINATION ${LIB_INSTALL_DIR}/limbo/cmake )