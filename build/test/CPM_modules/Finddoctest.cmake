include("/home/deb/.CPM/cpm/CPM_0.38.6.cmake")
CPMAddPackage("GITHUB_REPOSITORY;doctest/doctest;VERSION;2.4.9;EXCLUDE_FROM_ALL;YES;SYSTEM;YES;")
set(doctest_FOUND TRUE)