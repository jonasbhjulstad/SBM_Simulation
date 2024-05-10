function(assign_source_header_names dev_headers source_names header_names)
    set(SRC_LIST "")
    set(HDR_LIST "")
    foreach(DEV_HEADER ${dev_headers})
        #get path to dev_header relative to ${PROJECT_SOURCE_DIR}/dev_headers/
        
        file(RELATIVE_PATH RELATIVE_PATH ${PROJECT_SOURCE_DIR}/dev_headers/ ${DEV_HEADER})
        #remove extension
        string(REPLACE ".hpp" "" DEV_HEADER_NAME ${RELATIVE_PATH})


        set(SOURCE_NAME ${PROJECT_BINARY_DIR}/include/${DEV_HEADER_NAME}.cpp)
        set(HEADER_NAME ${PROJECT_BINARY_DIR}/include/${DEV_HEADER_NAME}.hpp)
        list(APPEND SRC_LIST ${SOURCE_NAME})
        list(APPEND HDR_LIST ${HEADER_NAME})
    endforeach()
    set(${source_names} "${SRC_LIST}" PARENT_SCOPE)
    set(${header_names} "${HDR_LIST}" PARENT_SCOPE)
endfunction()
function(create_if_not_exists files)
foreach(file ${files})
    if(NOT EXISTS ${file})
        file(WRITE ${file} "//auto generated file")
    endif()
    endforeach()
endfunction()
function(add_lzz_target target_name dev_headers output_dir source_names header_names)

    file(MAKE_DIRECTORY ${output_dir})
    set(HDR_LIST "")
    set(SRC_LIST "")
    assign_source_header_names("${dev_headers}" SRC_LIST HDR_LIST)
    create_if_not_exists("${SRC_LIST}")
    create_if_not_exists("${HDR_LIST}")
    add_custom_target(${target_name} DEPENDS ${HDR_LIST}
    COMMAND lzz -sx cpp -hx hpp -sl -il -hl -tl -hd -sd 
    -k ${target_name}
    ${dev_headers} 
    -I${PROJECT_SOURCE_DIR}/dev_headers/ 
    -o ${output_dir}
    SOURCES ${dev_headers}
    )
    set(${source_names} "${SRC_LIST}" PARENT_SCOPE)
    set(${header_names} "${HDR_LIST}" PARENT_SCOPE)
endfunction()

function(add_lzz_library library_name dev_directory)
file(GLOB DEV_HEADERS ${dev_directory}/*.hpp)
set(LZZ_SOURCE_FILES "")
set(LZZ_HEADER_FILES "")

file(RELATIVE_PATH DEV_REL_PATH ${PROJECT_SOURCE_DIR}/dev_headers/ ${dev_directory})
add_lzz_target("${library_name}_LZZ" "${DEV_HEADERS}" "${PROJECT_BINARY_DIR}/include/${DEV_REL_PATH}" LZZ_SOURCE_FILES LZZ_HEADER_FILES)
add_library(${library_name} ${LZZ_SOURCE_FILES} ${LZZ_HEADER_FILES})
add_dependencies(${library_name} "${library_name}_LZZ")

target_include_directories(${library_name} PUBLIC ${SIR_SBM_INCLUDE_DIR})
target_link_options(${library_name} PUBLIC "-fsycl")
endfunction()