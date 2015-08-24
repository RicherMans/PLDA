cmake_minimum_required(VERSION 2.6)
project(LDAKaldi)



set(COVAR_STATIC_LIB covarstats)

set(LDA_STATIC_LIB lda)

set(BINARY_DIR ${CMAKE_SOURCE_DIR}/bin/)

set(CHTK_DIR ${PROJECT_SOURCE_DIR}/chtk)

set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

set(EXECUTABLE_OUTPUT_PATH ${BINARY_DIR})

set(LIBRARY_OUTPUT_PATH ${BINARY_DIR})

#set(EXECUTABLE_NAME main)

#set(EXECUTABLE_PATH main.cpp)


#Add for KALDI the cmake module
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")
message("Looking for FindKaldi.cmake in " ${CMAKE_MODULE_PATH})


if (NOT KALDI_ROOT)
    set(KALDI_ROOT $ENV{KALDI_ROOT})
else(NOT KALDI_ROOT)
    message(FATAL_ERROR "KALDI_ROOT is not specified in the current environment, please run KALDI_ROOT=/YOUPATH/TO/KALDIROOT/")
endif()

if (NOT ATLAS_ROOT)
  set(ATLAS_ROOT ${KALDI_TOOLS_DIR}/ATLAS)
endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(ENV{ATLAS_DIR} ${ATLAS_ROOT})
  if (VERBOSE)
    find_package(Atlas REQUIRED)
  else ()
    find_package(Atlas QUIET REQUIRED)
  endif ()
else()
  set(ATLAS_INCLUDE_DIR ${ATLAS_ROOT}/include)
endif()


find_package(Kaldi REQUIRED)
find_package(PythonLibs REQUIRED)

add_definitions(-DDEBUG=0)
add_definitions(-DNO_OUTPUT=0)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -Wall -DKALDI_DOUBLEPRECISION=0 -DHAVE_POSIX_MEMALIGN -DHAVE_CXXABI_H -Winit-self -Wno-unused-local-typedefs -Wno-sign-compare -DHAVE_EXECINFO_H=1 -flax-vector-conversions")

set(CXX_BASE_FLAGS "-W -Wall -Werror -Wold-style-cast ")

if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    # Typical flags for Linux
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_ATLAS")

    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
else ()
      message(FATAL_ERROR "Barista does not support ${CMAKE_SYSTEM_NAME}.")
endif()


set(LIBRARY_OUTPUT_PATH ${BINARY_DIR})

include_directories(
    ${KALDI_SRC_DIR}
    ${PORTAUDIO_INCLUDE_DIR}
    ${OPENFST_INCLUDE_DIR}
    ${ATLAS_INCLUDE_DIR}
    ${CHTK_DIR}
    ${PYTHON_INCLUDE_DIRS}
    )


set(COVAR_HEADERS
    ${SOURCE_DIR}/CovarianceStats.hpp
    )

set(PYCOVAR_SOURCES 
    ${SOURCE_DIR}/ldamodule.cpp)

set(CHTK_SOURCES
    ${CHTK_DIR}/chtk.cpp)

set(COVAR_SOURCES
    ${SOURCE_DIR}/CovarianceStats.cpp
    )

add_library(${COVAR_STATIC_LIB} STATIC ${COVAR_HEADERS} ${COVAR_SOURCES})
target_link_libraries(${COVAR_STATIC_LIB} dl m pthread z)

add_library(chtk SHARED ${CHTK_SOURCES})
target_link_libraries(chtk z)

add_library(${LDA_STATIC_LIB} SHARED ${PYCOVAR_SOURCES})

target_link_libraries(${LDA_STATIC_LIB} dl m pthread chtk ${KALDI_LIBRARIES} ${OPENFST_LIBRARY} ${ATLAS_LIBRARIES} ${PYTHON_LIBRARIES} ${COVAR_STATIC_LIB})
#add_executable(${EXECUTABLE_NAME} ${EXECUTABLE_PATH})
#target_link_libraries(${EXECUTABLE_NAME} dl m pthread ${KALDI_LIBRARIES} ${OPENFST_LIBRARY} ${ATLAS_LIBRARIES} ${COVAR_STATIC_LIB} ${PYTHON_LIBRARIES})
