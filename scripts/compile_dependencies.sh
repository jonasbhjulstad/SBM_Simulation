function configure_with_sycl()
{
#get basename of $1
project_name=$(basename $1)
project_name_uppercase=$(echo $project_name | tr '[:lower:]' '[:upper:]')
    cmake -S $1 -B "$1/build/" -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -D${project_name_uppercase}_SYCL_TARGETS=$1 
}