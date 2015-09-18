# This file was adapted from the cppa find_package script in cppa-benchmarks.
# - Try to find libcppa
# Once done this will define
#
#  CPPA_FOUND        - system has libcppa
#  CPPA_INCLUDE_DIR  - libcppa include dir
#  CPPA_LIBRARY      - link againgst libcppa
#  CPPA_VERSION      - version in {major}.{minor}.{patch} format

if (CPPA_LIBRARY AND CPPA_INCLUDE_DIR)
  set(CPPA_FOUND TRUE)
else (CPPA_LIBRARY AND CPPA_INCLUDE_DIR)

  find_path(CPPA_INCLUDE_DIR
    NAMES
      cppa/cppa.hpp
    PATHS
      ${CPPA_ROOT}/include
      ${CPPA_ROOT}/libcppa
      /usr/include
      /usr/local/include
      /opt/local/include
      /sw/include
      ${CPPA_INCLUDE_PATH}
      ${CPPA_LIBRARY_PATH}
      ${CMAKE_INCLUDE_PATH}
      ${CMAKE_INSTALL_PREFIX}/include
      ../libcppa
      ../../libcppa
      ../../../libcppa
  )
  
  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    set(CPPA_BUILD_DIR build-gcc)
  elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(CPPA_BUILD_DIR build-clang)
  endif ()
  
  find_library(CPPA_LIBRARY
    NAMES
      libcppa_static
      cppa_static
    PATHS
      ${CPPA_ROOT}/lib
      ${CPPA_ROOT}/build/lib
      ${CPPA_ROOT}/libcppa/build/lib
      ${CPPA_ROOT}/${CPPA_BUILD_DIR}/lib
      ${CPPA_ROOT}/libcppa/${CPPA_BUILD_DIR}/lib
      /usr/lib
      /usr/local/lib
      /opt/local/lib
      /sw/lib
      ${CPPA_INCLUDE_PATH}
      ${CPPA_LIBRARY_PATH}
      ${CMAKE_LIBRARY_PATH}
      ${CMAKE_INSTALL_PREFIX}/lib
      ${LIBRARY_OUTPUT_PATH}
      ../libcppa/build/lib
      ../../libcppa/build/lib
      ../../../libcppa/build/lib
      ../libcppa/${CPPA_BUILD_DIR}/lib
      ../../libcppa/${CPPA_BUILD_DIR}/lib
      ../../../libcppa/${CPPA_BUILD_DIR}/lib
  )
  
  # extract CPPA_VERSION from config.hpp
  if (CPPA_INCLUDE_DIR)
    # we assume version 0.8.1 if CPPA_VERSION is not defined config.hpp
    set(CPPA_VERSION 801)
    file(READ "${CPPA_INCLUDE_DIR}/cppa/config.hpp" CPPA_CONFIG_HPP_CONTENT)
    string(REGEX REPLACE ".*#define CPPA_VERSION ([0-9]+).*" "\\1" CPPA_VERSION "${CPPA_CONFIG_HPP_CONTENT}")
    if ("${CPPA_VERSION}" MATCHES "^[0-9]+$")
      math(EXPR CPPA_VERSION_MAJOR "${CPPA_VERSION} / 100000")
      math(EXPR CPPA_VERSION_MINOR "${CPPA_VERSION} / 100 % 1000")
      math(EXPR CPPA_VERSION_PATCH "${CPPA_VERSION} % 100")
      set(CPPA_VERSION "${CPPA_VERSION_MAJOR}.${CPPA_VERSION_MINOR}.${CPPA_VERSION_PATCH}")
    else ()
      set(CPPA_VERSION "0.8.1")
    endif ()
    message (STATUS "libcppa version: ${CPPA_VERSION}")
  endif (CPPA_INCLUDE_DIR)
  
  
  if (CPPA_INCLUDE_DIR)
    if (NOT LibCPPA_FIND_QUIETLY)
      message (STATUS "libcppa include: ${CPPA_INCLUDE_DIR}")
    endif (NOT LibCPPA_FIND_QUIETLY)
  else (CPPA_INCLUDE_DIR)
    message (SEND_ERROR "libcppa header files NOT found.")
  endif (CPPA_INCLUDE_DIR)
  
  if (CPPA_LIBRARY) 
    if (NOT LibCPPA_FIND_QUIETLY)
      message (STATUS "libcppa library: ${CPPA_LIBRARY}")
    endif (NOT LibCPPA_FIND_QUIETLY)
  else (CPPA_LIBRARY)
    message (SEND_ERROR "libcppa static library not found. Make sure libcppa was configured with --build-static option.")
  endif (CPPA_LIBRARY)
  
  if (CPPA_INCLUDE_DIR AND CPPA_LIBRARY)
    set(CPPA_FOUND TRUE)
    set(CPPA_INCLUDE_DIR ${CPPA_INCLUDE_DIR})
    set(CPPA_LIBRARY ${CPPA_LIBRARY})
  endif (CPPA_INCLUDE_DIR AND CPPA_LIBRARY)

endif (CPPA_LIBRARY AND CPPA_INCLUDE_DIR)

if(LibCPPA_FIND_REQUIRED AND NOT CPPA_FOUND)
  message(FATAL_ERROR "Could not find libcppa.")
endif()
