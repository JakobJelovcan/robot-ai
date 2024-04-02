set(LIB_NAME llama-common)

set(SRC_Cpp
    common.cpp
    sampling.cpp
    grammar-parser.cpp
    unicode.cpp
    unicode-data.cpp
    llama.cpp
)
    
set(SRC_PublicHeaders
    unicode.h
    unicode-data.h
    llama.h
    common.h
    sampling.h
    log.h
    grammar-parser.h
)

prepend_include(${TARGET_FOLDER_NAME} SRC_PublicHeaders)

find_package(Threads REQUIRED)

add_library(${LIB_NAME} STATIC
            ${SRC_Cpp}
            ${SRC_PublicHeaders})

target_link_libraries(
    ${LIB_NAME} PRIVATE
                whisper
)

set_property(TARGET ${LIB_NAME} PROPERTY CXX_STANDARD 20)
set_property(TARGET ${LIB_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET ${LIB_NAME} PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
                                              ${CMAKE_CURRENT_BINARY_DIR}/../include
)