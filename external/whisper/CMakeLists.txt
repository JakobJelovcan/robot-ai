FetchContent_Declare(
    whisper
    GIT_REPOSITORY https://github.com/ggerganov/whisper.cpp.git
    GIT_TAG        1d7657f40974e251ea42275e155a8abfb24228ef
    GIT_PROGRESS   ON
    GIT_SHALLOW    
    EXCLUDE_FROM_ALL
    SYSTEM
)

FetchContent_MakeAvailable(whisper)