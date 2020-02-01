"""pymodpde.py: a symbolic module that generates the modified equation for time-dependent partial differential equation
based on the used finite difference scheme."""

__author__ = "Mokbel Karam , James C. Sutherland, and Tony Saad"
__copyright__ = "Copyright (c) 2019, Mokbel Karam"

__credits__ = ["University of Utah Department of Chemical Engineering"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Mokbel Karam"
__email__ = "mokbel.karam@chemeng.utah.edu"
__status__ = "Production"

from sympy import *
from itertools import product

i, j, k, n = symbols('i j k n')


class DifferentialEquation:
    def __init__(self, dependentVar: str, independentVars: list, indices: list = [i, j, k],
                 timeIndex: symbol.Symbol = n):
        '''
        Parameters:
            dependentVar (string): name of the dependent variable
            independentVars (list of string): names of the independent variables
            indices (list of symbols): symbols for the indices of the independent variables
            timeIndex (symbol): symbolic variable of the time index

        Examples:
            >>> DE = DifferentialEquation(dependentVar='u', independentVars=['x', 'y'], indices=[i, j], timeIndex=n)
        '''

        assert isinstance(dependentVar,
                          str), 'DifferentialEquation() parameter dependentVar={} not of <class "str">'.format(
            dependentVar)
        assert isinstance(independentVars,
                          list), 'independentVars() parameter independentVars={} not of <class "list">'.format(
            independentVars)
        assert isinstance(indices, list), 'indices() parameter indices={} not of <class "list">'.format(indices)
        assert isinstance(timeIndex,
                          symbol.Symbol), 'timeIndex() parameter timeIndex={} not of <class "sympy.core.symbol.Symbol">'.format(
            timeIndex)
        for indepVar in independentVars:
            assert isinstance(indepVar, str), 'independentVars members are not of <class "str">'.format(independentVars)
        for index in indices:
            assert isinstance(index,
                              symbol.Symbol), 'indices members are not of <class "sympy.core.symbol.Symbol">'.format(
                indices)

        if len(independentVars) > 3:
            raise Exception('No more than three independent variable is allowed!')
        else:
            self.__independentVars = independentVars
            self.__dependentVar_name = dependentVar

            self.__indices = indices
            self.__timeIndex = timeIndex

            self.__independent_vars()

            setattr(self, self.__dependentVar_name, self.dependent_var_func)
            self.indepVarsSym = [self.vars[var]['sym'] for var in self.__independentVars]
            self.indepVarsSym.append(self.t['sym'])
            self.dependentVar = Function(self.__dependentVar_name)(*self.indepVarsSym)

            self.latex_ME = {'lhs': '', 'rhs': {}}

            self.indicies = {}
            for var in self.__independentVars:
                self.indicies[var] = self.vars[var]['index']
            self.lhs = (self.dependent_var_func(self.t['index'] + 1, **self.indicies) - self.dependent_var_func(
                self.t['index'],
                **self.indicies)) / \
                       self.t['variation']
            self.rhs = None

    def get_independent_vars(self):
        '''
        Returns:
            self.__independentVars (list): list of independent variables names
        '''
        return self.__independentVars

    def __independent_vars(self):
        '''
        Defines the symbols for the independent variables, differential elements, wave number variables, and indices
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
            variationSymStr = '\Delta\ {}'.format(var)
            setattr(self, variationName, symbols(variationSymStr))
            self.vars[var]['variation'] = getattr(self, variationName)
            self.vars[var]['index'] = index
            num += 1
        self.t['sym'] = symbols('t')
        self.t['ampFactor'] = symbols('q')
        setattr(self, 'dt', symbols('\Delta{t}'))
        self.t['variation'] = getattr(self, 'dt')
        self.t['index'] = self.__timeIndex

    def dependent_var_func(self, time, **kwargs):
        '''
        The function assigned to the dependent variable name. It has the following form exp(alpha tn) exp(ikx) exp(iky) ...

        Parameters:
            time (symbolic expression): time step at which we are applying this function ex: n, n+1, n-1, ..., <timeIndex\> + number.
            kwargs (symbolic expression): the stencil points at which we are applying this function ex: x=i+3, y=j+1, ..., <independentVar\> = <spatialIndex\> + number

        Returns:
            symbolic expression of this function applied at time index and points

        Examples:

            >>> <DE>.<dependentVar>(time=n+1, x=i+1, y=j)

            the following example is about advection using Forward in Time and Upwind in Space (FTUS) scheme

            >>> i, j, n, a = symbols("i j n a")
            >>> DE = DifferentialEquation(dependentVar='u', independentVars=['x', 'y'], indices=[i, j], timeIndex=n)
            >>> advection = -a (DE.u(time=n, x=i, y=j) - DE.u(time=n, x=i-1, y=j))/DE.dx
            >>> DE.set_rhs(advection)
            >>> pretty_print(DE.modified_equation(nterms=2))

            another example where we change the name of the dependent variable from 'u' to 'f'
            >>> i, j, n, a = symbols("i j n a")
            >>> DE = DifferentialEquation(dependentVar='f', independentVars=['x', 'y'], indices=[i, j], timeIndex=n)
            >>> advection = -a (DE.f(time=n, x=i, y=j) - DE.f(time=n, x=i-1, y=j))/DE.dx
            >>> DE.set_rhs(advection)
            >>> pretty_print(DE.modified_equation(nterms=2))
        '''

        assert isinstance(time, add.Add) or isinstance(time,
                                                       symbol.Symbol), 'dependent_var_func() parameter time={} not of <class "sympy.core.add.Add">, or <class "sympy.core.symbol.Symbol">'.format(
            time)
        time_symbols = list(time.free_symbols)
        for sym in time_symbols:
            assert sym == self.t[
                'index'], 'dependent_var_func() parameter time={} inappropriate time index is used. Use {} instead.'.format(
                time, self.t['index'])

        keys = list(kwargs.keys())

        for var in keys:
            var_symbols = list(kwargs[var].free_symbols)
            assert len(
                var_symbols) == 1, 'dependent_var_func() parameter {}={} inappropriate number of indecies is used for {}'.format(
                var, kwargs[var], var)
            assert var_symbols[0] == self.vars[var][
                'index'], 'dependent_var_func() parameter {}={} other index is used for {}. Use {} index instead.'.format(
                var, kwargs[var], var, self.vars[var]['index'])

        expression = exp(self.t['ampFactor'] * (self.t['sym'] + (time - self.t['index']) * self.t['variation']))
        for var in keys:
            expression *= exp(1j * self.vars[var]['waveNum'] * (
                    self.vars[var]['sym'] + (kwargs[var] - self.vars[var]['index']) * self.vars[var]['variation']))
        return expression

    def __stencil_gen(self, points: list, order: int):
        '''
        Generates finite difference equation based on the location of sampled points and derivative order

        Parameters:
            points (list int): stencil of length N needed ex: [-1,0,1] stencil around 0
            order (int > 0): the order of derivatives d, d<N

        Returns:
             the finite difference coefficients along with the points used in a dictionary
                {'points':[],'coefs':[]}

        Examples:
            >>> <DE>.__stencil_gen(points=[-1,0],order=1)
        '''

        assert isinstance(points, list), '__stencil_gen() parameter points={} not of <class "list">'.format(points)
        for pt in points:
            assert isinstance(pt, int), 'elements of points={} are not of <class "int">'.format(points)
        assert order < len(points), 'Enter a derivative order that is less than the number of points in your stencil.'

        numPts = len(points)
        M = []
        for i in range(numPts):
            M.append([s ** i for s in points])
        M = Matrix(M)
        b = Matrix([factorial(order) * 1 if j == order else 0 for j in range(numPts)])
        coefs = list(M.inv() * b)
        return {'points': points, 'coefs': coefs}

    def expr(self, order, direction, time, stencil):
        '''
        Generates an expression based on the stencil, the direction,  order of the derivative, and the time at which the expression is evaluated.

        Parameters:
            order (int): order of the derivative
            direction (string): the name of the independent variable that indicate the direction of the derivative
            time (symbolic expression): time at which to evaluate the expression. ex: n+1 or n
            stencil (list of int): N points used for the stencil gen function

        Returns:
            symbolic expression

        Examples:
             >>> <DE>.expr(order=1, direction='x', time=n, stencil=[-1,0])
        '''

        assert isinstance(direction, str), 'exp() parameter direcction={} not of <class "str">'.format(direction)
        assert direction in self.__independentVars, 'direcction={} not an independent variable. indepVar={}'.format(
            direction, self.__independentVars)
        assert isinstance(time, add.Add) \
               or isinstance(time,symbol.Symbol), \
               'expr() parameter time={} not of <class "sympy.core.add.Add">,' \
               ' or <class "sympy.core.symbol.Symbol">'.format(time)

        time_symbols = list(time.free_symbols)
        for sym in time_symbols:
            assert sym == self.t[
                'index'], 'dependent_var_func() parameter time={} inappropriate time index is used. Use {} instead.'.format(
                time, self.t['index'])

        stencil = self.__stencil_gen(stencil, order)
        expression = 0
        for coef, pt in zip(stencil['coefs'], stencil['points']):

            kwargs = {}
            for var in self.__independentVars:
                if var == direction:
                    kwargs[var] = self.vars[direction]['index'] + pt
                else:
                    kwargs[var] = self.vars[var]['index']
            expression += coef * self.dependent_var_func(time=time, **kwargs) / (
                    self.vars[direction]['variation'] ** order)
        return ratsimp(expression)

    def modified_equation(self, nterms):
        '''
        Computes the values of the modified equation coefficients a_{ijk} where i, j and k represent
        the order of derivatives in the <indep var1\> , <indep var2\>, and <indep var3\> directions, respectively. These are written as
        a_ijk * u_{ijk}.

        Parameters:
            nterms (int):Number of terms to compute in the modified equation

        Returns:
             latex (string): Latex representation of the modified equation as ' lhs = rhs '

        Examples:
            >>> <DE>.modified_equation(nterms=2)
        '''

        assert nterms > 0, 'modified_equation() member nterms={} has to be greater than zero.'.format(nterms)

        q = self.__solve_amp_factor()
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
                coefs['a{}'.format(ies)] = nsimplify(coefficient)
                derivs['a{}'.format(ies)] = Derivative(self.dependentVar, *wrt_vars)

        me_lhs = Derivative(self.dependentVar, self.t['sym'], 1)
        me_rhs = 0
        self.latex_ME['lhs'] += latex(me_lhs)
        for key in coefs.keys():
            me_rhs += coefs[key] * derivs[key]
            self.latex_ME['rhs'][key[1:]] = latex(coefs[key] * derivs[key])
        self.ME = Eq(me_lhs, me_rhs)

        return self.__latex()

    def __solve_amp_factor(self):
        '''
        Solve for the amplification factor of the numerical discritazation of the partial differential equation

        Returns:
             (expression): symbolic expression of the rhs of the amplification factor
        '''
        A = symbols('A')
        # compute the amplification factor
        lhs1 = simplify(self.lhs / self.dependent_var_func(self.t['index'], **self.indicies))
        rhs1 = simplify(self.rhs / self.dependent_var_func(self.t['index'], **self.indicies))
        eq = lhs1 - rhs1
        eq = eq.subs(exp(self.t['ampFactor'] * self.t['variation']), A)
        eq = eq.subs(exp(self.t['variation'] * self.t['ampFactor']), A)
        eq = expand(eq)
        eq = collect(eq, A)
        logEqdt = simplify(solve(eq, A)[0])
        q = log(logEqdt) / self.t['variation']  # amplification factor
        return q

    def amp_factor(self):
        '''
        Creats the latex representation of the amplification factor

        Returns:
            latex (string): Latex representation of the amplification factor as ' lhs = rhs '
        '''
        lhs = symbols('alpha')
        rhs = self.__solve_amp_factor()
        eq = Eq(lhs,rhs)
        return latex(eq)

    def __latex(self):
        '''
        Returns:
            latex (string): Latex representation of the modified equation as ' lhs = rhs '

        '''
        strings = {}
        for key in self.latex_ME['rhs'].keys():
            num = sum([int(x) for x in [char for char in key]])
            string = self.latex_ME['rhs'][key]
            firstDelPos = string.rfind("{")
            secondDelPos = string.rfind("}")
            string = string.replace(string[firstDelPos:secondDelPos + 1], "")

            var_string = " " + string[-1] + " "
            string = string[:-1]
            rPartialPos = string.rfind("partial")
            varNewPos = string[:rPartialPos].rfind("}")
            string = string[:varNewPos] + var_string + string[varNewPos:]
            if num in list(strings.keys()):
                strings[num] += ' ' + string if string[0] == '-' else ' + ' + string
            else:
                strings[num] = ' ' + string if string[0] == '-' else ' + ' + string
        lhs_string = self.latex_ME['lhs']
        firstDelPos = lhs_string.rfind("{")
        secondDelPos = lhs_string.rfind("}")
        lhs_string = lhs_string.replace(lhs_string[firstDelPos:secondDelPos + 1], "")
        var_string = " " + lhs_string[-1] + " "
        lhs_string = lhs_string[:-1]
        rPartialPos = lhs_string.rfind("partial")
        varNewPos = lhs_string[:rPartialPos].rfind("}")
        lhs_string = lhs_string[:varNewPos] + var_string + lhs_string[varNewPos:]

        latex_str = lhs_string + ' = '
        for i in sorted(strings.keys()):
            latex_str += strings[i]
        return latex_str

    def __set_lhs(self):
        '''
        This function is not defined yet.
        '''
        raise Exception('For now we only support by default first order time derivative.')

    def set_rhs(self, expression):
        '''
        sets the rhs of the DifferentialEquation
        Parameters:
            expression (symbolic expression): linear combination of expression generated from <DE\>.expr(...) or <DE\>.<dependentVar\>(...)

        Examples:
            >>> DE = DifferentialEquation(dependentVar="u",independentVars =["x"])
            >>> a = symbols('a')

            using DE.expr(...)

            >>> advectionTerm = DE.expr(order=1,direction="x",time=n,stencil=[-1, 0])
            >>> DE.set_rhs(expression= - a * advectionTerm)

            or using  DE.<dependentVar\>(...)

            >>> advectionTerm = (DE.u(time=n, x=i) - DE.u(time=n, x=i-1))/DE.dx
            >>> DE.set_rhs(expression= - a * advectionTerm)

        '''

        assert not isinstance(expression, str), 'set_rhs() parameter expression={} not a symbolic expression'.format(
            expression)

        self.rhs = expression

    def __rhs(self):
        '''
        Returns:
             (expression):  the rhs of the differential equation
        '''
        return self.rhs

    def __lhs(self):
        '''
        Returns:
            (expression):  the lhs of the differential equation
        '''
        return self.lhs
