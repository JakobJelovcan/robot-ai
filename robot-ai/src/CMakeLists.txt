set(SRC_Cpp
    main.cpp
    whisper_wrapper.cpp
    llama_wrapper.cpp
)
    
set(SRC_PublicHeaders
    whisper_wrapper.hpp
    llama_wrapper.hpp
)

prepend_include(${TARGET_FOLDER_NAME} SRC_PublicHeaders)

find_package(Threads REQUIRED)

add_executable(${REPO_NAME}
    ${SRC_Cpp})

target_link_libraries(
    ${REPO_NAME} PRIVATE
                    daq::opendaq
                    ggml
                    whisper
                    llama
                    SDL2-static
                    Boost::program_options
                    Boost::algorithm
)

set_property(TARGET ${REPO_NAME} PROPERTY CXX_STANDARD 20)
set_property(TARGET ${REPO_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET ${REPO_NAME} PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(${REPO_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
                                              ${CMAKE_CURRENT_BINARY_DIR}/../include
)