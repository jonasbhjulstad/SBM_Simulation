file(GLOB_RECURSE SOURCES *.cpp)
add_library(${PROJECT_NAME} ${SOURCES})
default_configure_target(${PROJECT_NAME})
default_link_target(${PROJECT_NAME})