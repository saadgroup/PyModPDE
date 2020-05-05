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
import functools


try:
    from IPython import get_ipython
    if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
        raise ImportError("console")
    from IPython.display import display, Math, clear_output
except:
    pass

i, j, k, n = symbols('i j k n')


class DifferentialEquation:
    def __init__(self, dependentVarName: str, independentVarsNames: list, indices: list = [i, j, k],
                 timeIndex: symbol.Symbol = n):
        '''
        Parameters:
            dependentVarName (string): name of the dependent variable
            independentVarsNames (list of string): names of the independent variables
            indices (list of symbols): symbols for the indices of the independent variables
            timeIndex (symbol): symbolic variable of the time index

        Examples:
            >>> DE = DifferentialEquation(dependentVarName='u', independentVarsNames=['x', 'y'], indices=[i, j], timeIndex=n)
        '''

        assert isinstance(dependentVarName,
                          str), 'DifferentialEquation() parameter dependentVarName={} not of <class "str">'.format(
            dependentVarName)
        assert isinstance(independentVarsNames,
                          list), 'independentVarsNames() parameter independentVarsNames={} not of <class "list">'.format(
            independentVarsNames)
        assert isinstance(indices, list), 'indices() parameter indices={} not of <class "list">'.format(indices)
        assert isinstance(timeIndex,
                          symbol.Symbol), 'timeIndex() parameter timeIndex={} not of <class "sympy.core.symbol.Symbol">'.format(
            timeIndex)
        for indepVar in independentVarsNames:
            assert isinstance(indepVar, str), 'independentVarsNames members are not of <class "str">'.format(independentVarsNames)
        for index in indices:
            assert isinstance(index,
                              symbol.Symbol), 'indices members are not of <class "sympy.core.symbol.Symbol">'.format(
                indices)

        if len(independentVarsNames) > 3:
            raise Exception('No more than three independent variable is allowed!')
        else:
            self.__independentVars = independentVarsNames
            self.__dependentVar_name = dependentVarName

            self.__indices = indices
            self.__timeIndex = timeIndex

            self.__independent_vars()

            setattr(self, self.__dependentVar_name, self.dependent_var_func)
            self.indepVarsSym = [self.vars[var]['sym'] for var in self.__independentVars]
            self.indepVarsSym.append(self.t['sym'])
            self.dependentVar = Function(self.__dependentVar_name)(*self.indepVarsSym)

            self.__latex_ME = {'lhs': '', 'rhs': {}}

            self.indicies = {}
            for var in self.__independentVars:
                self.indicies[var] = self.vars[var]['index']
            self.lhs = (self.dependent_var_func(self.t['index'] + 1, **self.indicies) - self.dependent_var_func(
                self.t['index'],
                **self.indicies)) / \
                       self.t['variation']
            self.rhs = None

            self.__latex_amp_factor = None
            self.__ME = None
            self.__amp_factor = None
            self.__amp_factor_exponent = None
            self.__latex_amp_factor_exponent = None


    @property
    def symbolicModifiedEquation(self):
        '''
        Returns:
             the symbolic modified equation
        '''
        return self.__ME

    @property
    def latexModifiedEquation(self):
        '''
        Returns:
             the Latex string of the modified equation
        '''
        return self.__latex()

    @property
    def latexAmplificationFactor(self):
        '''
        Returns:
             the Latex string of the amplification factor
        '''
        return self.__latex_amp_factor

    @property
    def symbolicAmplificationFactor(self):
        '''
        Returns:
             the symbolic amplification factor
        '''
        return self.__amp_factor

    @property
    def symbolicAmplificationFactorExponent(self):
        '''
        Returns:
             the symbolic amplification factor exponent
        '''
        return self.__amp_factor_exponent

    @property
    def latexAmplificationFactorExponent(self):
        '''
        Returns:
             the Latex string of the amplification factor exponent
        '''
        return self.__latex_amp_factor_exponent

    @property
    def independentVars(self):
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
            variationSymStr = '\Delta{}'.format('{'+var+'}')
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
            >>> DE = DifferentialEquation(dependentVarName='u', independentVarsNames=['x', 'y'], indices=[i, j], timeIndex=n)
            >>> advection = -a (DE.u(time=n, x=i, y=j) - DE.u(time=n, x=i-1, y=j))/DE.dx
            >>> DE.set_rhs(advection)
            >>> pretty_print(DE.modified_equation(nterms=2))

            another example where we change the name of the dependent variable from 'u' to 'f'
            >>> i, j, n, a = symbols("i j n a")
            >>> DE = DifferentialEquation(dependentVarName='f', independentVarsNames=['x', 'y'], indices=[i, j], timeIndex=n)
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

    def expr(self, order, directionName, time, stencil):
        '''
        Generates an expression based on the stencil, the directionName,  order of the derivative, and the time at which the expression is evaluated.

        Parameters:
            order (int): order of the derivative
            directionName (string): the name of the independent variable that indicate the directionName of the derivative
            time (symbolic expression): time at which to evaluate the expression. ex: n+1 or n
            stencil (list of int): N points used for the stencil gen function

        Returns:
            symbolic expression

        Examples:
             >>> <DE>.expr(order=1, directionName='x', time=n, stencil=[-1,0])
        '''

        assert isinstance(directionName, str), 'exp() parameter direcction={} not of <class "str">'.format(directionName)
        assert directionName in self.__independentVars, 'direcction={} not an independent variable. indepVar={}'.format(
            directionName, self.__independentVars)
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
                if var == directionName:
                    kwargs[var] = self.vars[directionName]['index'] + pt
                else:
                    kwargs[var] = self.vars[var]['index']
            expression += coef * self.dependent_var_func(time=time, **kwargs) / (
                    self.vars[directionName]['variation'] ** order)
        return ratsimp(expression)

    def __printer(foo):
        '''
        decorator that prints the results based on where they are executed: jupyter or script
        '''
        @functools.wraps(foo)
        def Print(self, *args, **kwargs):
            try:
                if 'IPKernelApp' in get_ipython().config:  # pragma: no cover
                    if foo.__name__ == 'modified_equation':
                        foo(self, *args, **kwargs)
                        return display(Math(self.__latex()))
                    elif foo.__name__ in ['amp_factor','amp_factor_exponent'] :
                        return display(Math(latex(foo(self, *args, **kwargs))))
            except:
                symbolic_form = foo(self, *args, **kwargs)
                subs_dict = {}
                for indep in self.indepVarsSym:
                    subs_dict['\Delta{}'.format('{'+str(indep)+'}')] = var('d{}'.format(str(indep)))
                return pprint(symbolic_form.subs(subs_dict))


        return Print

    @__printer
    def modified_equation(self, nterms):
        '''
        Computes the values of the modified equation coefficients a_{ijk} where i, j and k represent
        the order of derivatives in the <indep var1\> , <indep var2\>, and <indep var3\> directions, respectively. These are written as
        a_ijk * u_{ijk}.

        Parameters:
            nterms (int): Number of in the modified equation. nterms is greater than zero.

        Returns:
             latex (display): Latex formatted representation of the modified equation as ' lhs = rhs ' in jupyter or console

        Examples:
            >>> <DE>.modified_equation(nterms=2)
        '''

        assert nterms > 0, 'modified_equation() member nterms={} has to be greater than zero.'.format(nterms)

        q = self.__solve_amp_exponent()

        order = self.__infer_order(q) # infering maximum order from the amplification factor.

        couples = (i for i in product(list(range(0, order + nterms)), repeat=len(self.__independentVars)) if
                   (sum(i) < order + nterms and sum(i) > 0))

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
        self.__latex_ME['lhs'] = latex(me_lhs)
        for key in coefs.keys():
            me_rhs += coefs[key] * derivs[key]
            self.__latex_ME['rhs'][key[1:]] = latex(coefs[key]) + ' ' + latex(derivs[key])
        self.__ME = Eq(me_lhs, me_rhs)
        return self.__ME

    def __infer_order(self, amp_factor):
        '''
        This function is used to infer the highest derivative order on  the rhs of the PDE using the amplification factor.
        this is done by counting the instances of differential elements and by searching for combinations of these elements
        in the amplification factor.
        :param amp_factor: symbolic expression of the amplification factor
        :return: (int) the order-derivative of the PDE's RHS.
        '''

        maximums = [0 for _ in range(
            len(self.__independentVars))]  # initiating a list with zeros based on the number of independent variables
        orders = []  # list that store the order each derivative with respect to one independent variable ( not for cross derivative)

        def rep(expr):
            '''
            Recursive function that traverse the symbolic tree searching for the differential elements (represents derivative order)
            and store the results in a list.
            :param expr: a symbolic expression
            :return: None
            '''
            base_expr = expr.as_base_exp()  # infer the exponents of the expression
            # if (len(base_expr) == 2 ) and (str(self.dx) == str(base_expr[0]) or str(self.dy) == str(base_expr[0])):
            if (len(base_expr) == 2) and any(list(
                    [str(self.vars[self.__independentVars[num]]['variation']) == str(base_expr[0]) for num in
                     range(len(self.__independentVars))])):
                # if the base expr is one of the differential elements store that into the orders list
                orders.append(base_expr)
                # print(expr.as_base_exp())
            for arg in expr.args:
                rep(arg)  # recursively call rep to transverse the amplification factor symbolic tree.

        rep(amp_factor)  # calling the function rep

        # in this for loop we go over all the values in orders and look for the maximum value of exponents
        # and store them in an organized way in maximums list
        for arg in orders:
            for num, var in enumerate(self.__independentVars):
                if arg[0] == self.vars[var]['variation']:
                    maximums[num] = max(maximums[num], abs(arg[1]))

                # print(amp_factor.has(self.vars[var]['variation']**maximums[num]))

        # checking for cross derivatives orders.
        ranges = [range(-max, max + 1) for max in maximums]
        # print(*ranges)
        products = list(product(*ranges))
        # print(list(products))

        comb_max = 0
        for p in products:
            var_comb = 1
            for num in range(len(maximums)):
                var_comb *= self.vars[self.__independentVars[num]]['variation'] ** p[num]

            if amp_factor.has(var_comb):
                comb_max = max(comb_max, sum([abs(p[i]) for i in range(len(p))]))

            # print('{}, {}'.format(var_comb,amp_factor.has(var_comb)))

        # choosing the maximum value for order between derivatives and cross derivatives.
        order = max(max(maximums), comb_max)

        # value for the maximum order on the rhs
        return order


    def __solve_amp_exponent(self):
        '''
        Solve for the amplification factor of the numerical discritazation of the partial differential equation

        Returns:
             (expression): symbolic expression of the rhs of the amplification factor
        '''
        e_alpha_dt = self.__solve_amp_factor()
        q = 1/self.t['variation'] * log(e_alpha_dt)  # alpha
        self.__amp_factor_exponent = q
        return q

    def __solve_amp_factor(self):
        A = symbols('A')
        # compute the amplification factor
        lhs1 = simplify(self.lhs / self.dependent_var_func(self.t['index'], **self.indicies))
        rhs1 = simplify(self.rhs / self.dependent_var_func(self.t['index'], **self.indicies))
        eq = lhs1 - rhs1
        eq = eq.subs(exp(self.t['ampFactor'] * self.t['variation']), A)
        eq = eq.subs(exp(self.t['variation'] * self.t['ampFactor']), A)
        eq = expand(eq)
        eq = collect(eq, A)
        e_alpha_dt = simplify(solve(eq, A)[0])
        return e_alpha_dt

    @__printer
    def amp_factor(self):
        '''
        Creats the latex representation of the amplification factor

        Returns:
            latex (display): Latex formatted representation of the amplification factor as ' lhs = rhs ' in jupyter or console
        '''
        lhs = exp(symbols('alpha')*self.t['variation'])
        rhs = self.__solve_amp_factor()
        self.__amp_factor = Eq(lhs,rhs)
        self.__latex_amp_factor = latex(self.__amp_factor)
        return self.__amp_factor

    @__printer
    def amp_factor_exponent(self):
        '''
        Creats the latex representation of the amplification factor exponent alpha

        Returns:
            latex (display): Latex formatted representation of the amplification factor exponent as ' lhs = rhs ' in jupyter or console
        '''
        lhs = symbols('alpha')
        rhs = self.__solve_amp_exponent()
        self.__amp_factor_exponent = Eq(lhs, rhs)
        self.__latex_amp_factor_exponent = latex(self.__amp_factor_exponent)
        return self.__amp_factor_exponent

    def __latex(self):
        '''
        Returns:
            latex (string): Latex representation of the modified equation as ' lhs = rhs '

        '''
        strings = {}
        for key in self.__latex_ME['rhs'].keys():
            num = sum([int(x) for x in [char for char in key]])
            string = self.__latex_ME['rhs'][key]
            firstDelPos = string.rfind("{")
            secondDelPos = string.rfind("}")
            string = string.replace(string[firstDelPos:secondDelPos + 1], "")
            var_string = " " + string[-1] + " "
            string = string[:-1]
            rPartialPos = string.rfind("partial")
            varNewPos = string[:rPartialPos].rfind("}{")
            string = string[:varNewPos] + var_string + string[varNewPos:]
            if num in list(strings.keys()):
                strings[num] += ' ' + string if string[0] == '-' else ' + ' + string
            else:
                strings[num] = ' ' + string if string[0] == '-' else ' + ' + string
        lhs_string = self.__latex_ME['lhs']
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
            >>> DE = DifferentialEquation(dependentVarName="u",independentVarsNames =["x"])
            >>> a = symbols('a')

            using DE.expr(...)

            >>> advectionTerm = DE.expr(order=1,directionName="x",time=n,stencil=[-1, 0])
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
