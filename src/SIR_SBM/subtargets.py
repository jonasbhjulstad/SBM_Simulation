from pathlib import Path
import os
import sys


def add_subdirs(subdirs, pwd):
    with open(pwd / "CMakeLists.txt", "w") as f:
        for subdir in subdirs:
            # make cmakelists.txt
            f.write("add_subdirectory({})".format(subdir.name))
            f.write("\n")


def link_library(cwd, target, libname, type="PRIVATE"):
    with open(cwd / target / "CMakeLists.txt", "a") as f:
        f.write("target_link_libraries(" + target + " PRIVATE " + libname + ")")
        f.write("\n")


if __name__ == '__main__':
    cwd = Path(__file__).parent
    # set cwd
    os.chdir(cwd)

    subdirs = [d for d in cwd.iterdir() if d.is_dir()]
    add_subdirs(subdirs, cwd)
    for subdir in subdirs:
        with open(subdir / 'CMakeLists.txt', 'w') as f:
            cpp_files = [
                sub.name for sub in subdir.iterdir() if sub.suffix == '.cpp']
            if subdir.name == "utils":
                cpp_files.append("${CMAKE_CURRENT_BINARY_DIR}/filepaths.cpp")
            subdirname = subdir.name
            f.write("add_library(" + subdirname + " " +
                    "\n\t".join([str(cpp) for cpp in cpp_files]) + ")")
            f.write("\n")
            f.write("target_include_directories(" +
                    subdirname + " PUBLIC ${PROJECT_SOURCE_DIR}/include)")
            f.write("\n")

    with open(cwd / "utils/CMakeLists.txt", "a") as f:
        f.write(
            "configure_file(${CMAKE_CURRENT_LIST_DIR}/filepaths.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/filepaths.cpp @ONLY)\n")

    sycl_targets = ["utils"]
    for sycl_target in sycl_targets:
        with open(cwd / "CMakeLists.txt", "a") as f:
            f.write("custom_configure_sycl(" + sycl_target + ")")
            f.write("\n")
    graph_targets = []
    cppiter_targets = ["graph"]
    yaml_targets = ["graph", "sycl"]
    casadi_targets = []
    onedpl_targets = ["random", "graph"]
    utils_targets = []
    dependencies = {"sycl": sycl_targets,
                    "graph": graph_targets,
                    "cppitertools::cppitertools": cppiter_targets,
                    "yaml-cpp::yaml-cpp": yaml_targets,
                    "casadi": casadi_targets,
                    "oneDPL": onedpl_targets,
                    "utils": utils_targets}

    # iterate over dependencies
    for target in dependencies:
        _ = [link_library(cwd, sub, target, type="PUBLIC")
             for sub in dependencies[target]]

        # main library
    with open(cwd / "CMakeLists.txt", "a") as f:
        f.write("add_library(SIR_SBM STATIC sir_sbm.cpp)\n")
        f.write(
            "target_include_directories(SIR_SBM PUBLIC ${PROJECT_SOURCE_DIR}/include)\n")
        f.write("target_link_libraries(SIR_SBM PRIVATE " +
                "\n\t\t\t\t".join([sub.name for sub in subdirs]) + ")\n")
