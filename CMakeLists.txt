cmake_minimum_required(VERSION 3.25)
set(REPO_NAME openDAQ-ai)
set(REPO_OPTION_PREFIX AI)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MESSAGE_CONTEXT ${REPO_NAME})
set(CMAKE_MESSAGE_CONTEXT_SHOW ON CACHE BOOL "Show CMake message context")

if (CMAKE_FOLDER)
    set(CMAKE_FOLDER ${CMAKE_FOLDER}/${REPO_NAME})
else()
    set(CMAKE_FOLDER ${REPO_NAME})
endif()

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(external)

project(openDAQ-ai
        LANGUAGES CXX
        VERSION 1.0.0
)

include(cmake/Utils.cmake)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wno-psabi)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -gdwarf-2 -sanitize=undefined")
endif()

add_subdirectory(${REPO_NAME})