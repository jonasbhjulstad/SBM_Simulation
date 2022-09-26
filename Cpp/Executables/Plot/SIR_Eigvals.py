import numpy as np
from numpy.linalg import eig
def SIR_jac(S, I, alpha, beta, N_pop):
    return np.array([[-beta/N_pop*I, beta/N_pop*I],[-beta/N_pop*S, beta/N_pop*S - alpha]])

if __name__ == '__main__':
    
    N_pop = 1000
    alpha = 1./9
    R0 = 2.0
    beta = R0*alpha    
    I0 = N_pop/10
    N_pop_max = 1000

    S = np.linspace(0, N_pop_max, 100)
    I = np.linspace(0, N_pop_max, 100)

    Sv, Iv = np.meshgrid(S, I)


    eigvals = [[SIR_jac(s, i, alpha, beta, N_pop)[0] for s, i in zip(sv, iv)] for sv, iv in zip(Sv, Iv)]
    print(eigvals)
