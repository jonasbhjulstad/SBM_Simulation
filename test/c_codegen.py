from casadi import *
if __name__ == '__main__':

    x = MX.sym('x', 2)
    y = MX.sym('y', 1)
    
    f = Function('f', [x, y], [x[0] + y[0], x[1] + y[0]])
    f.generate('f.c')
    