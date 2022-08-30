import casadi as cs

def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False

#Deconstructs a single pySINDY term into a list of tokens
def deconstruct_term(term):
    arr = []
    power_next = False
    if term[0].isdigit():
        return 1
    for letter in term:
        if power_next or letter.isdigit():
            arr.append(int(letter))
        else:
            if (letter == '^'):
                power_next = True
            elif letter != ' ':
                arr.append(letter)
    return arr

#constructs a single MX-term from a list of tokens
def construct_mx_term(X, U, term):
    symdict = {'x':X, 'u':U}
    symterm = 1
    sym = symdict[term[0]]
    idx = term[1]
    if (len(term) ==2):
        pow = 1
    else:
        pow = term[2]
    symterm*= cs.power(sym[idx], pow)
    return symterm

#Constructs MX-equations from deconstructed pySINDY-equations
def construct_mx_coeff(X, U, deconstructed_eq):    
    mx_term = 0
    mx_eq = 0
    for i in range(len(deconstructed_eq)):
        if (is_float(deconstructed_eq[i])):
            mx_eq += mx_term
            mx_term = deconstructed_eq[i]
        else:
            mx_term *= construct_mx_term(X, U, deconstruct_term(deconstructed_eq[i]))
        
    mx_eq += mx_term
    return mx_eq

#Deconstructs a pySINDY equation into a list of terms
def deconstruct_equation(eq):
    arr = []
    for term in eq.split('+'):
        for token in term.split(' '):
            if is_float(token):
                arr.append(float(token))
            elif token != '':
                arr.append(token)
    return arr


#Constructs MX-equations directly from a given pySINDY regression model
def construct_mx_equations(X, U, reg_model):
    deceqs = [deconstruct_equation(eq) for eq in reg_model.equations()]
    return [construct_mx_coeff(X, U, deceq) for deceq in deceqs]