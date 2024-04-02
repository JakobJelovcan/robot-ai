FetchContent_Declare(
    SDL
    GIT_REPOSITORY https://github.com/libsdl-org/SDL
    GIT_TAG        f1690e265e306818882c7c876a9e85492eeefa42
    GIT_PROGRESS   ON
    GIT_SHALLOW    
    EXCLUDE_FROM_ALL
    SYSTEM
)

FetchContent_MakeAvailable(SDL)