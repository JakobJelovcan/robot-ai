FetchContent_Declare(
    ggml
    GIT_REPOSITORY https://github.com/ggerganov/ggml.git
    GIT_TAG        8e413034b42e4fbedc2873166f61193b75f2622a
    GIT_PROGRESS   ON
    GIT_SHALLOW    
    EXCLUDE_FROM_ALL
    SYSTEM
)

FetchContent_MakeAvailable(ggml)