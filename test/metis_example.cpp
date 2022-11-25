#include <metis.h>
#include <cstddef>
int main()
{
    //demonstrate usage of metis
    idx_t nvtxs = 5;
    idx_t ncon = 1;
    idx_t objval;
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    idx_t xadj[] = {0, 2, 5, 8, 10, 12};
    idx_t adjncy[] = {1, 2, 0, 2, 3, 0, 1, 3, 4, 1, 2, 4};

    idx_t part[] = {0, 0, 0, 0, 0};
    METIS_PartGraphRecursive()
    return 0;

}