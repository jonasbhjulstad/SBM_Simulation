file(GLOB MIGRATION_HEADERS *.hpp)
#create a header containing all the headers
write_file("${CMAKE_CURRENT_SOURCE_DIR}/migration_headers.hpp" "#pragma once\n\n")
foreach(HEADER ${MIGRATION_HEADERS})
    write_file("${CMAKE_CURRENT_SOURCE_DIR}/migration_headers.hpp" "#include \"${HEADER}\"\n" APPEND)
endforeach()
