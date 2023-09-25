from SBM_Routines.Path_Config import *


if __name__ == '__main__':
    p_dirs = get_p_dirs(Data_dir)
    for p_dir in p_dirs:
        graph_dirs = get_graph_dirs(p_dir)

        for g_dir in graph_dirs:
