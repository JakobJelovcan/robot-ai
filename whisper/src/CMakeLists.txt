set(LIB_NAME whisper)

set(SRC_Cpp
    whisper.cpp
    common.cpp
    common-sdl.cpp
    common-ggml.cpp
)
    
set(SRC_PublicHeaders
    whisper.h
    common.h
    common-sdl.h
    common-ggml.h
)

prepend_include(${TARGET_FOLDER_NAME} SRC_PublicHeaders)

find_package(Threads REQUIRED)
find_package(SDL2 REQUIRED)

add_library(${LIB_NAME} STATIC
            ${SRC_Cpp}
            ${SRC_PublicHeaders})

target_link_libraries(
    ${LIB_NAME} PRIVATE
                ggml
                ${SDL2_LIBRARIES}
)

if (GGML_CUDA)
    add_compile_definitions(GGML_USE_CUDA)
elseif(GGML_CLBLAST)
    add_compile_definitions(GGML_USE_CLBLAST)
endif()

set_property(TARGET ${LIB_NAME} PROPERTY CXX_STANDARD 20)
set_property(TARGET ${LIB_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET ${LIB_NAME} PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
                                              ${CMAKE_CURRENT_BINARY_DIR}/../include
)