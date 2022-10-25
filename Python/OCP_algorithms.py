import casadi as cs
from pysindy_casadi_converter import construct_mx_equations
def quadratic_objective_solve(X_mean, U_mean, Wu, F_ODE, Nt, u_max = 0.1, u_min = 1e-6):
    Nx = X_mean.shape[1]
    Nu = 1

    X = cs.MX.sym('X', Nx, Nt)
    U = cs.MX.sym('U', Nu, Nt)

    g = []
    obj = 0

    gx = []
    obj = 0
    X_traj = [X_mean[0,:]]
    Xs = cs.MX.sym('Xs', Nx)
    obj = 0
    for i in range(Nt):
        if (i == 0):
            Xk = F_ODE(X_mean[0,:], U[i])
        else:
            Xk = F_ODE(Xk, U[i])
        X_traj.append(Xk)
        obj += Wu*(u_max - U[i])**2 + (Xk[1])**2
    get_X = cs.Function('get_X', [U], [cs.horzcat(*X_traj)])

    X0 = X_mean
    U0 = U_mean
    # prob = {'f': obj, 'x': W, 'g': g}
    # solver = cs.nlpsol('solver', 'ipopt', prob)
    lbx = [u_min]*Nt
    ubx = [u_max]*Nt
    # for i in range(Nt):
    #     ubx[1+3*i] = I_max
    # sol = solver(x0=cs.vertcat(X0[:], U_mean), lbx = lbw, ubx = ubw, lbg = lbg, ubg=ubg)
    prob = {'f': obj, 'x': U, 'g': []}
    solver = cs.nlpsol('solver', 'ipopt', prob)
    sol = solver(x0=U0, lbx = cs.vertcat(*lbx), ubx = cs.vertcat(*ubx))

    return (sol, get_X(sol['x'].full()), sol['x'].full())


def hospital_capacity_objective_solve(X_mean, U_mean, Wu, I_max, F_ODE,Nt,  u_max = 0.1, u_min = 1e-6):
    Nx = X_mean.shape[1]
    Nu = 1

    X = cs.MX.sym('X', Nx, Nt)
    U = cs.MX.sym('U', Nu, Nt)

    g = []
    obj = 0

    gx = []
    obj = 0
    X_traj = [X_mean[0,:]]
    Xs = cs.MX.sym('Xs', Nx)
    obj = 0
    for i in range(Nt):
        if (i == 0):
            Xk = F_ODE(X_mean[0,:], U[i])
        else:
            Xk = F_ODE(Xk, U[i])
        X_traj.append(Xk)
        obj += Wu*(u_max - U[i])**2 + (Xk[1])**2
        gx.append(Xk[1] - I_max)
    get_X = cs.Function('get_X', [U], [cs.horzcat(*X_traj)])

    X0 = X_mean
    U0 = U_mean
    lbx = [u_min]*Nt
    ubx = [u_max]*Nt
    ubg = [0]*Nt
    lbg = [0]*Nt
    # sol = solver(x0=cs.vertcat(X0[:], U_mean), lbx = lbw, ubx = ubw, lbg = lbg, ubg=ubg)
    prob = {'f': obj, 'x': U, 'g': cs.vertcat(*gx)}
    # prob = {'f': obj, 'x': U, 'g': g}
    Ng = len(gx)
    lbg = cs.DM.zeros(Ng)
    ubg = cs.DM.zeros(Ng)
    solver = cs.nlpsol('solver', 'ipopt', prob)
    sol = solver(x0=U0, lbx = cs.vertcat(*lbx), ubx = cs.vertcat(*ubx), ubg = ubg, lbg = lbg)

    return (sol, get_X(sol['x'].full()), sol['x'].full())