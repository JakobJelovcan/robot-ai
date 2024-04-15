set(SRC_Cpp
    whisper_wrapper.cpp
    llama_wrapper.cpp
)
    
set(SRC_PublicHeaders
    whisper_wrapper.hpp
    llama_wrapper.hpp
)

find_package(Threads REQUIRED)
find_package(SDL2 REQUIRED)

set(LIBS
    daq::opendaq
    ggml
    whisper
    llama
    ${SDL2_LIBRARIES}
    Boost::asio
    Boost::program_options
    Boost::algorithm)

prepend_include(${TARGET_FOLDER_NAME} SRC_PublicHeaders)

# Robot AI

add_executable(${REPO_NAME}
    robot_ai.cpp
    ${SRC_Cpp})

target_link_libraries(
    ${REPO_NAME} PRIVATE ${LIBS}
)

set_property(TARGET ${REPO_NAME} PROPERTY CXX_STANDARD 20)
set_property(TARGET ${REPO_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET ${REPO_NAME} PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(${REPO_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
                                              ${CMAKE_CURRENT_BINARY_DIR}/../include
)

add_dependencies(${REPO_NAME} daq::opcua_client_module)
add_dependencies(${REPO_NAME} daq::native_stream_cl_module)

# whisper test

add_executable(whisper_test
    whisper_test.cpp
    ${SRC_Cpp})

target_link_libraries(
    whisper_test PRIVATE ${LIBS}
)

set_property(TARGET whisper_test PROPERTY CXX_STANDARD 20)
set_property(TARGET whisper_test PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET whisper_test PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(whisper_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
                                              ${CMAKE_CURRENT_BINARY_DIR}/../include
)

# llama test

add_executable(llama_test
    llama_test.cpp
    ${SRC_Cpp})

target_link_libraries(
    llama_test PRIVATE ${LIBS}
)

set_property(TARGET llama_test PROPERTY CXX_STANDARD 20)
set_property(TARGET llama_test PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET llama_test PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(llama_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
                                              ${CMAKE_CURRENT_BINARY_DIR}/../include
)

# tts

add_executable(tts
    tts.cpp)

target_link_libraries(
    tts PRIVATE daq::opendaq
)

set_property(TARGET tts PROPERTY CXX_STANDARD 20)

# combined test

add_executable(combined_test
    combined_test.cpp
    ${SRC_Cpp})

target_link_libraries(
    combined_test PRIVATE ${LIBS}
)

set_property(TARGET combined_test PROPERTY CXX_STANDARD 20)
set_property(TARGET combined_test PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET combined_test PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(combined_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
                                              ${CMAKE_CURRENT_BINARY_DIR}/../include
)
