#default cmakelists
cmake_minimum_required(VERSION 3.22)

#c++20
set(CMAKE_CXX_STANDARD 20)
project(SIR_SBM)
set(SIR_SBM_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/)
set(SIR_SBM_LIBRARIES "")
add_compile_options("-fsycl")
add_compile_options("-fcolor-diagnostics")
option(SIR_SBM_USE_GPU "Use GPU" OFF)

find_program(iwyu_path NAMES include-what-you-use iwyu REQUIRED)
include(cmake/external_packages.cmake)
include(cmake/default_targets.cmake)
#export compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#make dir
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include/SIR_SBM/)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/src/SIR_SBM/)
set(SIR_SBM_DEV_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/dev_headers/SIR_SBM/)
set(SIR_SBM_INCLUDE_DIR ${PROJECT_BINARY_DIR}/include/)

set(SIR_SBM_LIBRARY_INCLUDE_DIRS ${SIR_SBM_INCLUDE_DIR} ${SYCL_INCLUDE_DIR} /usr/include/c++/11/)
file(GLOB DEV_HEADERS ${SIR_SBM_DEV_INCLUDE_DIR}/*.hpp)
#get ${PROJECT_BINARY_DIR}/include/SIR_SBM/dev_header_name.hpp/.cpp names
set(SIR_SBM_SOURCE_FILES "")
set(SIR_SBM_HEADER_FILES "")
foreach(DEV_HEADER ${DEV_HEADERS})
    #get basename no ext
    get_filename_component(DEV_HEADER_NAME ${DEV_HEADER} NAME_WE)
    set(SOURCE_NAME ${CMAKE_BINARY_DIR}/include/SIR_SBM/${DEV_HEADER_NAME}.cpp)
    set(HEADER_NAME ${CMAKE_BINARY_DIR}/include/SIR_SBM/${DEV_HEADER_NAME}.hpp)
    list(APPEND SIR_SBM_SOURCE_FILES ${SOURCE_NAME})
    list(APPEND SIR_SBM_HEADER_FILES ${HEADER_NAME})
    set_source_files_properties(${SOURCE_NAME} OBJECT_DEPENDS ${DEV_HEADER})
    set_source_files_properties(${HEADER_NAME} OBJECT_DEPENDS ${DEV_HEADER})
endforeach()

#create files if they do not exist (in order to satisfy cmake library target)
foreach(SOURCE_FILE ${SIR_SBM_SOURCE_FILES})
    if(NOT EXISTS ${SOURCE_FILE})
        file(WRITE ${SOURCE_FILE} "//auto generated file")
    endif()
endforeach()

foreach(HEADER_FILE ${SIR_SBM_HEADER_FILES})
    if(NOT EXISTS ${HEADER_FILE})
        file(WRITE ${HEADER_FILE} "//auto generated file")
    endif()
endforeach()

add_custom_target(DEV_HEADERS_target DEPENDS ${DEV_HEADERS})

#LZZ generation
add_custom_target(SIR_SBM_LZZ_Generation DEPENDS ${DEV_HEADERS}
COMMAND lzz -sx cpp -hx hpp -sl -il -hl -tl -hd -sd ${DEV_HEADERS} 
-I${PROJECT_SOURCE_DIR}/dev_headers/ 
-o ${CMAKE_BINARY_DIR}/include/SIR_SBM/
SOURCES ${DEV_HEADERS}
)
add_dependencies(SIR_SBM_LZZ_Generation DEV_HEADERS_target)
add_library(SIR_SBM ${SIR_SBM_SOURCE_FILES})
add_dependencies(SIR_SBM SIR_SBM_LZZ_Generation)
option(SIR_SBM_BUFFER_LOG "Enable buffer logging" OFF)
target_compile_definitions(SIR_SBM PUBLIC SIR_SBM_BUFFER_LOG=${SIR_SBM_BUFFER_LOG})
#make SIR_SBM depend on the generated files
target_include_directories(SIR_SBM PUBLIC ${SIR_SBM_INCLUDE_DIR})
target_link_options(SIR_SBM PUBLIC "-fsycl")

add_subdirectory(test)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)