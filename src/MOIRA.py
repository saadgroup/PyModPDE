from sympy import *
from itertools import product
from sympy.utilities.lambdify import lambdify, implemented_function

i, j, k, n = symbols('i j k n')


class DifferentialEquation:
    def __init__(self, dependentVar, independentVars, indices, timeIndex):
        '''
        :param dependentVar: string of the dependent variable
        :param independentVars: list of strings of the independent variables
        :param indices: list of symbolic variables of the indices used for the location of the independent variables
        :param timeIndex: symbolic variable for the time index
        '''
        self.__independentVars = independentVars
        self.__dependentVar_name = dependentVar

        self.__indices = indices
        self.__timeIndex = timeIndex

        self.__independent_vars()

        setattr(self, self.__dependentVar_name, self.function)
        self.indepVarsSym = [self.vars[var]['sym'] for var in self.__independentVars]
        self.indepVarsSym.append(self.t['sym'])
        self.dependentVar = Function(self.__dependentVar_name)(*self.indepVarsSym)

        self.latex_ME = {'lhs': '', 'rhs': {}}

    def get_independent_vars(self):
        return self.__independentVars

    def __independent_vars(self):
        self.vars = {}
        self.t = {}
        num = 1
        for var, index in zip(self.__independentVars, self.__indices):
            self.vars[var] = {}
            varName = 'indepVar{}'.format(num)
            setattr(self, varName, symbols(var))
            self.vars[var]['sym'] = getattr(self, varName)
            waveNumName = 'k{}'.format(num)
            setattr(self, waveNumName, symbols(waveNumName))
            self.vars[var]['waveNum'] = getattr(self, waveNumName)
            variationName = 'd{}'.format(var)
            setattr(self, variationName, symbols(variationName))
            self.vars[var]['variation'] = getattr(self, variationName)
            self.vars[var]['index'] = index
            num += 1
        self.t['sym'] = symbols('t')
        self.t['ampFactor'] = symbols('q')
        setattr(self, 'dt', symbols('dt'))
        self.t['variation'] = getattr(self, 'dt')
        self.t['index'] = self.__timeIndex

    def function(self, time, **kwargs):
        '''
        a function of the form exp(alpha tn) exp(ikx) exp(iky) ...
        :param tn: time step at which we are applying this function ex: n, n+1, n-1 ...
        :param kwargs: the stencile points at which we are applying this function ex: x=i+3, y=j+1 ...
        :return: a symbolic expression of this function applied at time tn and points <indep var1>,<indep var2> ...
        '''
        keys = list(kwargs.keys())
        expression = exp(self.t['ampFactor'] * (self.t['sym'] + (time - self.t['index']) * self.t['variation']))
        for var in keys:
            expression *= exp(1j * self.vars[var]['waveNum'] * (
                    self.vars[var]['sym'] + (kwargs[var] - self.vars[var]['index']) * self.vars[var]['variation']))
        return expression

    def stencil_gen(self, points, order):
        '''
        calling stencil generation function:   stencil_gen([0,1],1,u,t,dt)
        :param points: stencil of length N needed ex: [-1,0,1] stencil around 0
        :param order: the order of derivatives d, d<N
        :return: return the finite difference coefficients along with the points used in a dictionaray
                {'points':[],'coefs':[]}
        '''
        numPts = len(points)
        M = []
        for i in range(numPts):
            M.append([s ** i for s in points])
        M = Matrix(M)
        b = Matrix([factorial(order) * 1 if j == order else 0 for j in range(numPts)])
        coefs = list(M.inv() * b)
        return {'points': points, 'coefs': coefs}

    def expr(self, points, direction, order, time):
        '''
        :param points: list of points of size N used for the stencil generation
        :param direction: string, with the name of the independent variable that indicate the direction of the derivative
        :param order: int, order of the derivative
        :param time: symbolic expression, time at which to evaluate the expression. ex: n+1 or n
        :return: a symbolic expression
        '''
        points = points
        direction = direction
        order = order
        time = time
        stencil = self.stencil_gen(points, order)
        expression = 0
        for coef, pt in zip(stencil['coefs'], stencil['points']):

            kwargs = {}
            for var in self.__independentVars:
                if var == direction:
                    kwargs[var] = self.vars[direction]['index'] + pt
                else:
                    kwargs[var] = self.vars[var]['index']
            expression += coef * self.function(time=time, **kwargs) / (self.vars[direction]['variation'] ** order)
        return ratsimp(expression)

    def modified_equation(self, nterms):
        '''
        computes the values of the modified equation coefficients a_{ijk} where i, j and k represent
        the order of derivatives in the <indep var1> , <indep var2>, and <indep var3> directions, respectively. These are written as
        a_ijk * u_{ijk}.
        :param nterms: Number of terms to compute in the modified equation
        :return: bool, true if finished without error, false otherwise
        '''
        try:
            A = symbols('A')
            # compute the amplification factor
            lhs1 = simplify(self.lhs / self.function(self.t['index'], **self.indicies))
            rhs1 = simplify(self.rhs / self.function(self.t['index'], **self.indicies))
            eq = lhs1 - rhs1
            eq = eq.subs(exp(self.t['ampFactor'] * self.t['variation']), A)
            eq = eq.subs(exp(self.t['variation'] * self.t['ampFactor']), A)
            eq = expand(eq)
            eq = collect(eq, A)
            logEqdt = simplify(solve(eq, A)[0])
            q =  log(logEqdt) / self.t['variation']  # amplification factor
            couples = [i for i in product(list(range(0, nterms + 1)), repeat=len(self.__independentVars)) if
                       (sum(i) <= nterms and sum(i) > 0)]

            coefs = {}
            derivs = {}
            for couple in couples:
                wrt_vars = []
                wrt_wave_num = []
                waveNum = {}
                fac = 1
                N = 0
                ies = ''

                for num, var in enumerate(self.__independentVars):
                    wrt_wave_num.append(self.vars[var]['waveNum'])
                    waveNum[self.vars[var]['waveNum']] = 0
                    wrt_wave_num.append(couple[num])
                    wrt_vars.append(self.vars[var]['sym'])
                    wrt_vars.append(couple[num])
                    N = sum(couple)
                    fac *= factorial(couple[num])
                    ies += str(couple[num])

                diff_ = diff(q, *wrt_wave_num).subs(waveNum)
                frac = ratsimp(1 / (fac * I ** N))
                coefficient = simplify(frac * diff_)
                if coefficient != 0:
                    coefs['a{}'.format(ies)] = coefficient
                    derivs['a{}'.format(ies)] = Derivative(self.dependentVar, *wrt_vars)

            me_lhs = Derivative(self.dependentVar, self.t['sym'], 1)
            me_rhs = 0
            self.latex_ME['lhs'] += latex(me_lhs)
            for key in coefs.keys():
                me_rhs += coefs[key] * derivs[key]
                self.latex_ME['rhs'][key[1:]] = latex(nsimplify(coefs[key] * derivs[key]))
            self.ME = Eq(me_lhs, nsimplify(me_rhs))
            return True
        except:
            return False

    def latex(self):
        '''
        :return: string, the latex representation of the modified equation as ' lhs = rhs '
        '''
        strings = {}
        for key in self.latex_ME['rhs'].keys():
            num = sum([int(x) for x in [char for char in key]])
            string = self.latex_ME['rhs'][key]
            if num in list(strings.keys()):
                strings[num] += ' ' + string if string[0] == '-' else ' + ' + string
            else:
                strings[num] = ' ' + string if string[0] == '-' else ' + ' + string
        latex_str = self.latex_ME['lhs'] + ' = '
        for i in sorted(strings.keys()):
            latex_str += strings[i]
        return latex_str


class HyperbolicDE(DifferentialEquation):
    '''
    Derived of the parent class 'Differential equation'. this class define set_rhs function and set the lhs to be
    1rst order derivative in time
    '''

    def __init__(self, independentVars, dependentVar, indices=[i, j, k], timeIndex=n):
        '''
        :param dependentVar: string of the dependent variable
        :param independentVars: list of strings of the independent variables
        :param indices: list of symbolic variables of the indices used for the location of the independent variables has default as [i, j, k]
        :param timeIndex: symbolic variable for the time index has default as n
        '''
        if len(independentVars) > 3:
            raise Exception(msg='no more than three independent variable')
        else:
            super().__init__(dependentVar, independentVars, indices, timeIndex)
            self.indicies = {}
            for var in self.get_independent_vars():
                self.indicies[var] = self.vars[var]['index']
            self.lhs = (super().function(self.t['index'] + 1, **self.indicies) - super().function(self.t['index'],
                                                                                                 **self.indicies)) / \
                       self.t['variation']
            # self.lhs = super().function( 1, **self.indicies) - super().function(0,**self.indicies)/self.t['variation']
            self.rhs = None

    def set_rhs(self, expression):
        '''
        set the rhs of the HyperbolicDE
        :param expression: symbolic expression ( linear combination of expression generated from self.expr )
        '''
        self.rhs = expression

    def rhs(self):
        '''
        :return: expression for the rhs of the differential equation
        '''
        return self.rhs

    def lhs(self):
        '''
        :return: expression for the lhs of the differential equation
        '''
        return self.lhs
