set(SRC_Cpp
    main.cpp
    common.cpp
    common-sdl.cpp
    common-ggml.cpp
    grammar-parser.cpp
)
    
set(SRC_PublicHeaders
    main.hpp
    common.h
    dr_wav.h
    common-sdl.h
    common-ggml.h
    grammar-parser.h
)

prepend_include(${TARGET_FOLDER_NAME} SRC_PublicHeaders)

find_package(Threads REQUIRED)
find_package(SDL2 REQUIRED)

add_executable(${REPO_NAME}
    ${SRC_Cpp})

target_link_libraries(
    ${REPO_NAME} PRIVATE
                    daq::opendaq
                    whisper
                    Boost::program_options
                    Boost::algorithm
                    ${SDL2_LIBRARIES}
)

set_property(TARGET ${REPO_NAME} PROPERTY CXX_STANDARD 20)
set_property(TARGET ${REPO_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET ${REPO_NAME} PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(${REPO_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
                                              ${CMAKE_CURRENT_BINARY_DIR}/../include
)