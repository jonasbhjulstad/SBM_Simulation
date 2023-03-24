#!/bin/sh
source /opt/intel/oneapi/setvars.sh --include-intel-llvm
/opt/intel/oneapi/debugger/2023.0.0/gdb/intel64/bin/gdb-oneapi $@