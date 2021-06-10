cmake_minimum_required (VERSION 3.0.2)
project(limbo VERSION 2.1 DESCRIPTION "limbo LIbrary for Model-Based Optimization")

# Cpp standard
set(CMAKE_CXX_STANDARD 14)

# # Add library
# Interface because it's header only
add_library(limbo INTERFACE IMPORTED GLOBAL)

# Install
include(GNUInstallDirs)

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