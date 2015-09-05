cmake_minimum_required(VERSION 2.6)
project(LDAKaldi)


set(PACKAGE_VERSION_MAJOR 0)
set(PACKAGE_VERSION_MINOR 1)
set(PACKAGE_VERSION_PATCH 0)
set(PACKAGE_VERSION ${PACKAGE_VERSION_MAJOR}.${PACKAGE_VERSION_MINOR}.${PACKAGE_VERSION_PATCH})

set(LDA_SHARED_LIB lda)

set(LDA_STATIC_LIB static_lda)

set(PYTHONDIR ${CMAKE_SOURCE_DIR}/python)

set(MODULE_NAME liblda)

set(BINARY_DIR ${PYTHONDIR}/${MODULE_NAME}/)

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
find_package(NumPy REQUIRED)

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

add_subdirectory(python)

include_directories(
    ${KALDI_SRC_DIR}
    ${PORTAUDIO_INCLUDE_DIR}
    ${OPENFST_INCLUDE_DIR}
    ${ATLAS_INCLUDE_DIR}
    ${CHTK_DIR}
    ${PYTHON_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIRS}
    ${SOURCE_DIR}
    )

set(LDA_MAIN 
    ${SOURCE_DIR}/ldamodule.cpp)

set(LDA_HEADERS 
    ${SOURCE_DIR}/LDA.hpp)

set(LDA_SOURCES 
    ${SOURCE_DIR}/LDA.cpp)

set(CHTK_SOURCES
    ${CHTK_DIR}/chtk.cpp)

add_library(chtk SHARED ${CHTK_SOURCES})
target_link_libraries(chtk z)

add_library(${LDA_STATIC_LIB} SHARED ${LDA_SOURCES} ${LDA_HEADERS})
target_link_libraries(${LDA_STATIC_LIB} ${KALDI_LIBRARIES} ${OPENFST_LIBRARY} ${ATLAS_LIBRARIES})
#Error while linking occurs if we do not include that to the static library compile flags 
#IF( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64"  )
    #SET_TARGET_PROPERTIES(${LDA_STATIC_LIB} PROPERTIES COMPILE_FLAGS "-fPIC")
#ENDIF(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64"  )

add_library(${LDA_SHARED_LIB} SHARED ${LDA_MAIN} ${LDA_HEADERS})

#target_link_libraries(${LDA_SHARED_LIB} dl m pthread chtk ${PYTHON_LIBRARIES} ${KALDI_LIBRARIES} ${OPENFST_LIBRARY} ${ATLAS_LIBRARIES})
target_link_libraries(${LDA_SHARED_LIB} ${LDA_STATIC_LIB} dl m pthread chtk ${PYTHON_LIBRARIES})
#add_executable(${EXECUTABLE_NAME} ${EXECUTABLE_PATH})
#target_link_libraries(${EXECUTABLE_NAME} dl m pthread ${KALDI_LIBRARIES} ${OPENFST_LIBRARY} ${ATLAS_LIBRARIES} ${COVAR_STATIC_LIB} ${PYTHON_LIBRARIES})
