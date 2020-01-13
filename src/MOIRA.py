"""MOIRA.py: a symbolic module that generates the modified equation for time-dependent partial differential equation
based on the used finite difference scheme."""

__author__     = "Mokbel Karam , James C. Sutherland, and Tony Saad"
__copyright__  = "Copyright (c) 2019, Mokbel Karam"

__credits__    = ["University of Utah Department of Chemical Engineering"]
__license__    = "MIT"
__version__    = "1.0.0"
__maintainer__ = "Mokbel Karam"
__email__      = "mokbel.karam@chemeng.utah.edu"
__status__     = "Production"

from sympy import *
from itertools import product

i, j, k, n = symbols('i j k n')


class DifferentialEquation:
    def __init__(self, dependentVar, independentVars, indices=[i, j, k], timeIndex=n):
        '''
        Parameters:
            dependentVar (string): the dependent variable name
            independentVars (list of string): the independent variables names
            indices (list of symbols): symbols for the indices of the independent variables
            timeIndex (symbol): symbolic variable for the time index

        Examples:
            >>> DE = DifferentialEquation(independentVars=['x', 'y'], dependentVar='u', indices=[i, j], timeIndex=n)
        '''
        if len(independentVars) > 3:
            raise Exception('No more than three independent variable is allowed!')
        else:
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

            self.indicies = {}
            for var in self.__independentVars:
                self.indicies[var] = self.vars[var]['index']
            self.lhs = (self.function(self.t['index'] + 1, **self.indicies) - self.function(self.t['index'],
                                                                                                 **self.indicies)) / \
                       self.t['variation']
            self.rhs = None

    def get_independent_vars(self):
        '''
        Returns:
            self.__independentVars: a list of independent variables strings
        '''
        return self.__independentVars

    def __independent_vars(self):
        '''
        Define the symbols for the independent variables, differential elements, wave number variables, and indices
        '''
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
            variationSymStr= '\Delta\ {}'.format(var)
            setattr(self, variationName, symbols(variationSymStr))
            self.vars[var]['variation'] = getattr(self, variationName)
            self.vars[var]['index'] = index
            num += 1
        self.t['sym'] = symbols('t')
        self.t['ampFactor'] = symbols('q')
        setattr(self, 'dt', symbols('\Delta{t}'))
        self.t['variation'] = getattr(self, 'dt')
        self.t['index'] = self.__timeIndex

    def function(self, time, **kwargs):
        '''
        This is the function assigned to the dependent variable name. it has the following form exp(alpha tn) exp(ikx) exp(iky) ...

        Parameters:
            time (symbolic expression): time step at which we are applying this function ex: n, n+1, n-1 ...
            kwargs (symbolic expression): the stencil points at which we are applying this function ex: x=i+3, y=j+1 ...

        Returns:
            symbolic expression of this function applied at time tn and points <indep var1>,<indep var2> ...

        Examples:
            >>> <DE>.<dependentVar>(time=n+1, x=i+1, y=j)
        '''
        keys = list(kwargs.keys())
        expression = exp(self.t['ampFactor'] * (self.t['sym'] + (time - self.t['index']) * self.t['variation']))
        for var in keys:
            expression *= exp(1j * self.vars[var]['waveNum'] * (
                    self.vars[var]['sym'] + (kwargs[var] - self.vars[var]['index']) * self.vars[var]['variation']))
        return expression

    def stencil_gen(self, points, order):
        '''
        Generates finite difference equation based on the location of sampled points and derivative order

        Parameters:
            points (list int): stencil of length N needed ex: [-1,0,1] stencil around 0
            order (int > 0): the order of derivatives d, d<N

        Returns:
             the finite difference coefficients along with the points used in a dictionary
                {'points':[],'coefs':[]}

        Examples:
            >>> <DE>.stencil_gen(points=[-1,0],order=1)
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
        Generates an expression based on the stencil points, the direction,  order of the derivative, and the time at which the expression is evaluated.

        Parameters:
            points (list of int): N points used for the stencil gen function
            direction (string): the name of the independent variable that indicate the direction of the derivative
            order (int): order of the derivative
            time (symbolic expression): time at which to evaluate the expression. ex: n+1 or n

        Returns:
            symbolic expression

        Examples:
             >>> <DE>.expr(points=[-1,0],direction='x',order=1,time=n)
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
        Computes the values of the modified equation coefficients a_{ijk} where i, j and k represent
        the order of derivatives in the <indep var1> , <indep var2>, and <indep var3> directions, respectively. These are written as
        a_ijk * u_{ijk}.

        Parameters:
            nterms (int):Number of terms to compute in the modified equation

        Returns:
             bool: true if finished without error, false otherwise

        Examples:
            >>> <DE>.modified_equation(nterms=2)
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
                me_rhs += nsimplify(coefs[key]) * derivs[key]
                self.latex_ME['rhs'][key[1:]] = latex(nsimplify(coefs[key]) * derivs[key])
            self.ME = Eq(me_lhs, me_rhs)
            return True
        except:
            return False

    def latex(self):
        '''
        Returns:
            latex (string): the latex representation of the modified equation as ' lhs = rhs '

        Examples:
            >>> <DE>.latex()

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

    def set_lhs(self):
        '''
        This function is not defined yet.
        '''
        raise Exception('For now we only support by default first order time derivative.')

    def set_rhs(self, expression):
        '''
        set the rhs of the HyperbolicDE
        Parameters:
            expression (symbolic expression): linear combination of expression generated from <DE>.expr(...) or <DE>.<dependentVar>(...)

        Examples:
            >>> DE = HyperbolicDE(dependentVar="u",independentVars =["x"])
            >>> a = symbols('a')
            #using DE.expr(...)
            >>> advectionTerm = DE.expr(points=[-1, 0],  direction="x", order160=1, time=n)
            >>> DE.set_rhs(expression= - a * advectionTerm)
            #or using  DE.<dependentVar>(...)
            >>> advectionTerm = (DE.u(tn=n, x=i) - DE.u(tn=n, x=i-1))/DE.dx
            >>> DE.set_rhs(expression= - a * advectionTerm)

        '''
        self.rhs = expression

    def rhs(self):
        '''
        Returns:
             (expression):  the rhs of the differential equation
        '''
        return self.rhs

    def lhs(self):
        '''
        Returns:
            (expression):  the lhs of the differential equation
        '''
        return self.lhs
