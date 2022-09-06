import sys
BINDER_DIR = "/home/arch/Documents/Bernoulli_Network_Optimal_Control/Cpp/build/Binders/"
sys.path.insert(0, BINDER_DIR)
from pyFROLS import single_response_regression, feature_names
import pandas as pd
from glob import glob
import numpy as np

def print_regression_data(rd):
    for best_feature in rd:
        print("Index:\t{}".format(best_feature.index) + "\tERR:\t{}".format(best_feature.ERR) + "\tg:\t{}".format(best_feature.g))
    return

if __name__ == '__main__':

    DATA_DIR = "/home/arch/Documents/Bernoulli_Network_Optimal_Control/Cpp/data/"
    fnames = glob(DATA_DIR + "*.csv")

    dfs = [pd.read_csv(fname, delimiter=",") for fname in fnames[1:10] if "Quantile" not in fname]

    X_df = [df.loc[:, ['S', 'I', 'R', 'p_I']].to_numpy() for df in dfs]
    t = dfs[0]['t']
    X = np.vstack(X_df[:-1])
    Y = np.vstack(X_df[1:])[:,:-1]
    ERR_tol = 1e-3
    rd = single_response_regression(X, Y[:,0], ERR_tol)
    print_regression_data(rd)
    feature_names = feature_names(4, 3, 3)

    for best_feature in rd:
        print(feature_names[best_feature.index])

    # [print(r'{}'.format(name)) for name in feature_names]
    
    a = 1