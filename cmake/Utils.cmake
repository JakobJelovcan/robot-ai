function(prepend_include SUBFOLDER SOURCE_FILES)
    list(TRANSFORM ${SOURCE_FILES} PREPEND "../include/${SUBFOLDER}/")
    set( ${SOURCE_FILES} ${${SOURCE_FILES}} PARENT_SCOPE )
endfunction()

function(get_current_folder_name OUTFOLDER)
    get_filename_component(FOLDER ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    set(${OUTFOLDER} ${FOLDER} PARENT_SCOPE)
endfunction()