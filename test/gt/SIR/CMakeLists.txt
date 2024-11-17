add_library(gt_SIR SHARED gt_SIR.cpp)
graphtool_configure(gt_SIR)
set_target_properties(gt_SIR PROPERTIES POSITION_INDEPENDENT_CODE ON)