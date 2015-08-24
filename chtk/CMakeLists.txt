CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

project(CHTK)

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

option(ENABLE_STATIC "Build static (.a) library" ON)

add_library(chtk SHARED "chtk.cpp")
target_link_libraries(chtk z)
install(TARGETS "chtk" LIBRARY DESTINATION lib PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

if(ENABLE_STATIC)
    add_library(chtk-static STATIC "chtk.cpp")
    set_target_properties(chtk-static PROPERTIES OUTPUT_NAME "chtk")
    install(TARGETS "chtk-static" ARCHIVE DESTINATION lib)
endif(ENABLE_STATIC)

install(FILES "chtk.h" DESTINATION include)

