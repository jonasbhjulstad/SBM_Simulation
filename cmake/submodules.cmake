
set(${PROJECT_NAME}_SUBMODULE_DIR ${PROJECT_SOURCE_DIR}/submodules/)
set(${PROJECT_NAME}_SUBMODULES Static_RNG Dataframe SBM_Graph Buffer_Routines SBM_Database)
# message(WARNING "submodule_dir is ${${PROJECT_NAME}_SUBMODULE_DIR}")
foreach(submodule ${${PROJECT_NAME}_SUBMODULES})
#if ${submodule}_DIR is not defined
if(NOT DEFINED ${submodule}_DIR)
    set(${submodule}_DIR "${${PROJECT_NAME}_SUBMODULE_DIR}/${submodule}/build/" CACHE STRING "Build directory of ${submodule}")
endif()
if(NOT DEFINED ${submodule}_SOURCE_DIR)
    set(${submodule}_SOURCE_DIR "${${PROJECT_NAME}_SUBMODULE_DIR}/${submodule}/" CACHE STRING "Source directory of ${submodule}")
endif()
if(NOT DEFINED ${submodule}_BUILD_DIR)
    file(MAKE_DIRECTORY "${${PROJECT_NAME}_BINARY_DIR}/submodules/${submodule}/")
    set(${submodule}_BUILD_DIR "${${PROJECT_NAME}_BINARY_DIR}/submodules/${submodule}/" CACHE STRING "Binary directory of ${submodule}")
endif()
endforeach()



function(custom_submodule_add name)
CPMAddPackage(NAME ${name} SOURCE_DIR ${${name}_SOURCE_DIR} 
OPTIONS 
"${${name}_EXPORT} ON"
"ENABLE_TESTING OFF"
"${name}_BUILD_BINDER OFF")
endfunction()