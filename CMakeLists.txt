cmake_minimum_required(VERSION 3.17.2...3.29)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C Fortran)

find_package(Python COMPONENTS Development.Module NumPy REQUIRED)

# F2PY include path
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import numpy.f2py; print(numpy.f2py.get_include())"
  OUTPUT_VARIABLE F2PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# fortranobject object library
add_library(fortranobject OBJECT "${F2PY_INCLUDE_DIR}/fortranobject.c")
target_link_libraries(fortranobject PUBLIC Python::NumPy)
target_include_directories(fortranobject PUBLIC "${F2PY_INCLUDE_DIR}")
set_property(TARGET fortranobject PROPERTY POSITION_INDEPENDENT_CODE ON)

# Auto-generate wrappers
add_custom_command(
  OUTPUT bicubicmodule.c bicubic-f2pywrappers.f
  DEPENDS lensit/bicubic/bicubic.f90
  COMMAND "${Python_EXECUTABLE}" -m numpy.f2py
          "${CMAKE_CURRENT_SOURCE_DIR}/lensit/bicubic/bicubic.f90"
          -m bicubic --lower
  VERBATIM)

# Compile extension in the right namespace
python_add_library(
  bicubic MODULE
  "${CMAKE_CURRENT_BINARY_DIR}/bicubicmodule.c"
  "${CMAKE_CURRENT_BINARY_DIR}/bicubic-f2pywrappers.f"
  "${CMAKE_CURRENT_SOURCE_DIR}/lensit/bicubic/bicubic.f90"
  WITH_SOABI)

target_link_libraries(bicubic PRIVATE fortranobject)


# Automatically install to correct package dir
install(TARGETS bicubic
        LIBRARY DESTINATION lensit/bicubic)

message(STATUS "Installing to: ${CMAKE_INSTALL_PREFIX}")
