file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
if(NOT TARGET ${PROJECT_NAME})
# ---- Create binary ----

add_executable(${PROJECT_NAME} ${sources})

target_link_libraries(${PROJECT_NAME} PUBLIC ${PARENT_PROJECT_NAME}::${PARENT_PROJECT_NAME} doctest::doctest)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
configure_file(test_config.hpp.in test_config.hpp)
doctest_discover_tests(${PROJECT_NAME})
# enable compiler warnings
if(NOT TEST_INSTALLED_VERSION)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
# target_compile_options(SBM_Database PUBLIC -Wall -Wpedantic -Wextra -Werror)
elseif(MSVC)
target_compile_options(SBM_Database PUBLIC /W4 /WX)
target_compile_definitions(${PROJECT_NAME} PUBLIC DOCTEST_CONFIG_USE_STD_HEADERS)
endif()
endif()
endif()
