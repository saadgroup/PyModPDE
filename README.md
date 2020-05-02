# PyModPDE

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/saadgroup/PyModPDE/master?filepath=examples.ipynb)(press launch binder to run the examples.ipynb)

PyModPDE is a Python software that generates the modified equation for a first order time dependent partial differential equation (PDE) of the form
<br/>
<img src="http://latex.codecogs.com/svg.latex?u_t&space;=&space;\alpha_1&space;u_x&space;&plus;&space;\alpha_2&space;u_xx&space;&plus;&space;\ldots&space;=&space;\sum_1^M&space;\alpha_n&space;\frac{\partial&space;^n&space;u}{\partial&space;x^n}" title="u_t = \alpha_1 u_x + \alpha_2 u_xx + \ldots = \sum_n=1^M \alpha_n \frac{\partial ^n u}{\partial x^n}"/>
<br/>
Given a certain finite difference scheme for the PDE, 
<br/>
<img src="http://latex.codecogs.com/svg.latex?\delta_{t}u&plus;\delta_{x}u&space;&plus;&space;&plus;\delta_{xx}u&space;&plus;\ldots&space;=0" title="\delta_{t}u+\delta_{x}u + +\delta_{xx}u +\ldots =0" />
<br/>
PyModPDE computes the amplification factor of the finite difference scheme and also returns the the modified equation in the form
<br/>
<img src="http://latex.codecogs.com/svg.latex?u_{t}=a_{1}u_{x}&plus;a_{2}u_{xx}&plus;a_{3}u_{xxx}&plus;\cdots=\sum_1^P&space;a_{n}\frac{\partial^{n}u}{\partial&space;x^{n}}." title="u_{t}=a_{1}u_{x}+a_{2}u_{xx}+a_{3}u_{xxx}+\cdots=\sum_n=1^P a_{n}\frac{\partial^{n}u}{\partial x^{n}}." />
<br/>
where, in general, there are inifinitely more terms in the modified equation compared to the original PDE (P > M).

# Usage
To use PyModPDE, one has to instantiate a first order time dependent differential equation object by calling its constructor that has the following signature
```Python
DifferentialEquation(dependentVar,independentVars,indices=[i, j, k], timeIndex=n)
```
Once the user constructs an object of type `DifferentialEquation`, the next step is to start constructing the right-hand-side RHS for this equation. Two methods are available to achieve this: the first is to use the member function `expr` which has the following signature
```Python
expr(order, direction, time, stencil)
```
The second method is to use the dependent variable `name`, the `indices`, and the differential elements of the independent variable defined by the user, `d<independentVars>`. The signature of the member function `<dependentVars>` is as follow
```Python
<dependentVar>(time, **kwargs)
```
`time` here is the discrete time at which the expression of the dependent variable is evaluated ex: `n+1, n, ...` . `kwargs`are indicies of the spatial points at which this expression is evaluated, ex: `x=i+1, y=j, ...`.

After the construction of the rhs expression using one the two previous methods or a combination of both the user can set the rhs for this differential equation by calling the member function `set_rhs` that have the following signature
```Python
set_rhs(expression)
```
where `expression` is a symbolic expression of the rhs constructed using the previously described methods.

Now the member function, `modified_equation(...)`, can be used to generate the modified equation up to certain number of terms that the user specify, and return the modified equation in latex form. The signature of `modified_equation` is as follow
```Python
modified_equation(nterms)
```
where `nterms` is a positive integer that indicates the total number of terms in the modified equation.

# Sample code snippets
Below are some examples of using the PyModPDE software.
Starting with an example of using `expr(...)` with the advection equation in one dimension using Forward Euler for time discretization and UPWIND for spatial discretization

```Python
from src.pymodpde import DifferentialEquation, i, n 

# defining the advection velocity
a= symbols('a') 

#constructing a time dependent differential equation
DE = DifferentialEquation(dependentVar='u',independentVars=['x']) 

# method I of constructing the rhs:
advectionTerm1 = DE.expr(order=1,  direction='x', time=n, stencil=[-1, 0]) 

# setting the rhs of the differential equation
DE.set_rhs(- a * advectionTerm1 )

# displaying the amplification factor
DE.amp_factor()

# computing and displaying the modified equation up to two terms
DE.modified_equation(nterms=2)

```

Similarly, one can use the `<dependentVar>(...)` instead of `expr(...)` to construct the discretization of the rhs 

```Python
from src.pymodpde import DifferentialEquation, i, n 

# defining the advection velocity
a= symbols('a') 

#constructing a time dependent differential equation
DE = DifferentialEquation(dependentVar='u',independentVars=['x']) 

# method II of constructing the rhs:
advectionTerm = (DE.u(time=n, x=i) - DE.u(time=n, x=i-1))/DE.dx 

# setting the rhs of the differential equation
DE.set_rhs(- a * advectionTerm )

# displaying the amplification factor
DE.amp_factor()

# computing and displaying the modified equation up to two terms
DE.modified_equation(nterms=2)
```

---
## See Also

* [Documentation](https://htmlpreview.github.io/?https://github.com/saadgroup/PyModPDE/blob/master/doc/pymodpde.html)
--- 
