FetchContent_Declare(
    openDAQ
    GIT_REPOSITORY https://github.com/openDAQ/openDAQ.git
    GIT_TAG        0219fe4cffe28acbff4185021518d2fbf6292d71
    PATCH_COMMAND  git reset --hard && git apply ${CMAKE_CURRENT_LIST_DIR}/openDAQ.patch
    GIT_PROGRESS   ON
    GIT_SHALLOW    
    EXCLUDE_FROM_ALL
    SYSTEM
)

FetchContent_MakeAvailable(openDAQ)