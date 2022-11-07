if(${SYCL_GRAPH_USE_INTEL_SYCL})
    # get_property(importTargets DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
    # get_property(importTargetsAfter DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
    find_package(IntelDPCPP REQUIRED HINTS "/opt/intel/oneapi/compiler/latest/linux/IntelDPCPP")
    set(TBB_DIR "/opt/intel/oneapi/tbb/latest/lib/cmake/tbb
/")
    # find_package(TBB REQUIRED)

    # find_package(oneDPL REQUIRED HINTS "/opt/intel/oneapi/dpl/2021.7.0/lib/cmake/oneDPL/")

    # set(ONEAPI_LIBRARIES oneDPL TBB::tbb)

    # list(REMOVE_ITEM importTargetsAfter ${importTargets})
    # message(WARNING "${importTargetsAfter}")
    # set(CMAKE_CXX_COMPILER /opt/intel/oneapi/compiler/2022.1.0/linux/bin/icpx)
    # set(CMAKE_C_COMPILER /opt/intel/oneapi/compiler/2022.1.0/linux/bin/icx)
    set(SYCL_GRAPH_USE_SYCL ON)
endif()