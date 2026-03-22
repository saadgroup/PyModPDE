"""
pymodpde.py — A symbolic module for modified equation analysis of
finite difference schemes applied to time-dependent PDEs.

Based on the Chang (1990) / Karam et al. (2020) amplification-factor approach:
    1. Substitute the von-Neumann ansatz into the discrete scheme → amplification factor G = e^{αΔt}
    2. Solve for α (the amplification exponent)
    3. Differentiate α w.r.t. wave numbers at k=0 to recover modified-equation coefficients

Authors : Mokbel Karam, James C. Sutherland, Tony Saad (original)
Version  : 2.0.0 — cleaner internals, caching, no wildcard imports. Public API identical to v1.
License  : MIT
"""

from __future__ import annotations

__author__    = "Mokbel Karam, James C. Sutherland, and Tony Saad"
__copyright__ = "Copyright (c) 2019, Mokbel Karam"
__credits__   = ["University of Utah Department of Chemical Engineering"]
__license__   = "MIT"
__version__   = "2.0.0"
__maintainer__ = "Mokbel Karam"
__email__     = "mokbel.karam@chemeng.utah.edu"
__status__    = "Production"

from itertools import product
from typing import Any

import sympy as sp

# ── module-level symbols (same as v1) ─────────────────────────────────────────
i, j, k, n = sp.symbols("i j k n")

# also expose sympy.symbols so   from pymodpde_v2 import symbols   works
symbols = sp.symbols


# ── helpers ───────────────────────────────────────────────────────────────────

def _factorial_product(tup: tuple[int, ...]) -> sp.Integer:
    result = sp.Integer(1)
    for m in tup:
        result *= sp.factorial(m)
    return result


def _stencil_coefficients(points: list[int], order: int) -> list[sp.Rational]:
    """Finite-difference weights via Vandermonde solve (exact rationals)."""
    if order >= len(points):
        raise ValueError(
            f"Derivative order ({order}) must be less than the number of "
            f"stencil points ({len(points)})."
        )
    n_pts = len(points)
    M = sp.Matrix([[sp.Rational(s**row) for s in points] for row in range(n_pts)])
    rhs = sp.Matrix(
        [sp.factorial(order) if row == order else sp.Integer(0) for row in range(n_pts)]
    )
    return list(M.solve(rhs))


# ── main class ────────────────────────────────────────────────────────────────

class DifferentialEquation:
    """
    Symbolic representation of a first-order-in-time linear PDE discretised
    by a finite difference scheme.

    API is identical to v1 — all original argument names are preserved.

    Examples
    --------
    >>> from pymodpde import DifferentialEquation, symbols, i, n
    >>> a = symbols('a')
    >>> DE = DifferentialEquation(dependentVarName='u', independentVarsNames=['x'])
    >>> DE.set_rhs(-a * DE.expr(order=1, directionName='x', time=n, stencil=[-1, 0]))
    >>> DE.generate_modified_equation(nterms=3)
    >>> DE.display_modified_equation()
    """

    def __init__(
        self,
        dependentVarName: str,
        independentVarsNames: list[str],
        indices: list[sp.Symbol] | None = None,
        timeIndex: sp.Symbol | None = None,
    ) -> None:
        """
        Parameters
        ----------
        dependentVarName:
            Name of the dependent variable, e.g. ``'u'``.
        independentVarsNames:
            Names of the spatial independent variables, e.g. ``['x']`` or ``['x', 'y']``.
            At most three are supported.
        indices:
            Symbolic spatial grid indices (default: ``[i, j, k]`` trimmed to length).
        timeIndex:
            Symbolic time index (default: ``n``).
        """
        if not isinstance(dependentVarName, str):
            raise TypeError(
                f'DifferentialEquation() parameter dependentVarName={dependentVarName!r}'
                ' not of <class "str">'
            )
        if not isinstance(independentVarsNames, list) or not all(
            isinstance(v, str) for v in independentVarsNames
        ):
            raise TypeError(
                f'independentVarsNames={independentVarsNames!r} must be a list of strings'
            )
        if len(independentVarsNames) == 0 or len(independentVarsNames) > 3:
            raise ValueError("No more than three independent variables are allowed.")

        if timeIndex is None:
            timeIndex = n
        if not isinstance(timeIndex, sp.Symbol):
            raise TypeError(
                f'timeIndex={timeIndex!r} not of <class "sympy.core.symbol.Symbol">'
            )

        default_indices = [i, j, k]
        if indices is None:
            indices = default_indices[: len(independentVarsNames)]
        if len(indices) != len(independentVarsNames):
            raise ValueError("len(indices) must equal len(independentVarsNames).")
        for idx in indices:
            if not isinstance(idx, sp.Symbol):
                raise TypeError(f"indices members must be sympy Symbols, got {type(idx)}")

        self._dep_name: str = dependentVarName
        self._indep_names: list[str] = independentVarsNames
        self._indices: list[sp.Symbol] = indices
        self._time_idx: sp.Symbol = timeIndex

        self._setup_symbols()

        # expose DE.u(...), DE.f(...) etc. — same as v1 dependent_var_func
        setattr(self, self._dep_name, self.dependent_var_func)

        idx_dict = {v: self.vars[v]["index"] for v in self._indep_names}
        self.lhs: sp.Expr = (
            self.dependent_var_func(self._time_idx + 1, **idx_dict)
            - self.dependent_var_func(self._time_idx, **idx_dict)
        ) / self.t["variation"]
        self.rhs: sp.Expr | None = None

        # cached results
        self._amp_factor: sp.Eq | None = None
        self._amp_exponent: sp.Expr | None = None
        self._modified_eq: sp.Eq | None = None
        self._me_coefs: dict[str, sp.Expr] = {}
        self._me_derivs: dict[str, sp.Expr] = {}
        self._latex_me: str = ""

        self.__simplification_tolerance: float = 1e-6
        self._is_jupyter: bool = self._detect_jupyter()

    # ── symbol setup (mirrors v1 self.vars / self.t layout) ──────────────────

    def _setup_symbols(self) -> None:
        self.vars: dict[str, dict[str, Any]] = {}
        num = 1
        for var, spatial_idx in zip(self._indep_names, self._indices):
            self.vars[var] = {
                "sym":       sp.Symbol(var),
                "waveNum":   sp.Symbol(f"k{num}"),
                "variation": sp.Symbol(rf"\Delta{{{var}}}"),
                "index":     spatial_idx,
            }
            setattr(self, f"d{var}", self.vars[var]["variation"])
            num += 1

        self.t: dict[str, Any] = {
            "sym":       sp.Symbol("t"),
            "ampFactor": sp.Symbol("q"),
            "variation": sp.Symbol(r"\Delta{t}"),
            "index":     self._time_idx,
        }
        setattr(self, "dt", self.t["variation"])

        space_time = [self.vars[v]["sym"] for v in self._indep_names] + [self.t["sym"]]
        self.dependentVar = sp.Function(self._dep_name)(*space_time)

        # convenience list used by latex builder
        self.indepVarsSym = [self.vars[v]["sym"] for v in self._indep_names]
        self.indepVarsSym.append(self.t["sym"])

    # ── Jupyter detection ────────────────────────────────────────────────────

    @staticmethod
    def _detect_jupyter() -> bool:
        try:
            from IPython import get_ipython  # type: ignore
            cfg = get_ipython()
            return cfg is not None and "IPKernelApp" in cfg.config
        except Exception:
            return False

    # ── von-Neumann ansatz (v1 name: dependent_var_func) ─────────────────────

    def dependent_var_func(self, time: sp.Expr, **kwargs: sp.Expr) -> sp.Expr:
        """
        Von-Neumann wave ansatz:  e^{q·(t + (time-n)·Δt)} · ∏ e^{i·kₛ·(xₛ + offset·Δxₛ)}

        This is the function assigned to ``DE.u``, ``DE.f``, etc.

        Examples
        --------
        >>> DE.u(time=n, x=i+1)
        >>> DE.u(time=n+1, x=i, y=j-1)
        """
        expr = sp.exp(
            self.t["ampFactor"] * (
                self.t["sym"] + (time - self.t["index"]) * self.t["variation"]
            )
        )
        for var, grid_idx in kwargs.items():
            v = self.vars[var]
            expr *= sp.exp(
                sp.I * v["waveNum"] * (
                    v["sym"] + (grid_idx - v["index"]) * v["variation"]
                )
            )
        return expr

    # ── stencil / expr ───────────────────────────────────────────────────────

    def expr(
        self,
        order: int,
        directionName: str,
        time: sp.Expr,
        stencil: list[int],
    ) -> sp.Expr:
        """
        Build the finite-difference approximation of the *order*-th spatial
        derivative in *directionName* using *stencil* at discrete *time*.

        Parameters
        ----------
        order:         Derivative order (≥ 1).
        directionName: Spatial variable name, must be in ``independentVarsNames``.
        time:          Discrete time level, e.g. ``n`` or ``n+1``.
        stencil:       Integer offsets, e.g. ``[-1, 0]`` for upwind.
        """
        if directionName not in self._indep_names:
            raise ValueError(
                f"directionName='{directionName}' not in "
                f"independentVarsNames={self._indep_names}"
            )
        coefs = _stencil_coefficients(stencil, order)
        delta = self.vars[directionName]["variation"]
        idx0  = self.vars[directionName]["index"]

        total: sp.Expr = sp.Integer(0)
        for coef, pt in zip(coefs, stencil):
            kw = {v: self.vars[v]["index"] for v in self._indep_names}
            kw[directionName] = idx0 + pt
            total += coef * self.dependent_var_func(time=time, **kw)

        return sp.ratsimp(total / delta**order)

    # ── fluent RHS setter ────────────────────────────────────────────────────

    def set_rhs(self, expression: sp.Expr) -> "DifferentialEquation":
        """Set the right-hand side and return *self* for optional chaining."""
        if isinstance(expression, str):
            raise TypeError("expression must be a sympy Expr, not a string")
        self.rhs = expression
        # invalidate cached results
        self._amp_factor    = None
        self._amp_exponent  = None
        self._modified_eq   = None
        return self

    # ── amplification factor ─────────────────────────────────────────────────

    def _compute_amp_factor_rhs(self) -> sp.Expr:
        if self.rhs is None:
            raise RuntimeError("Call set_rhs() before generating the amplification factor.")
        A    = sp.Symbol("A")
        base = self.dependent_var_func(
            self.t["index"],
            **{v: self.vars[v]["index"] for v in self._indep_names}
        )
        lhs_norm = sp.simplify(self.lhs / base)
        rhs_norm = sp.simplify(self.rhs / base)
        eq = sp.expand(lhs_norm - rhs_norm)
        eq = eq.subs(sp.exp(self.t["ampFactor"] * self.t["variation"]), A)
        eq = sp.collect(eq, A)
        solutions = sp.solve(eq, A)
        if not solutions:
            raise RuntimeError(
                "Could not solve for the amplification factor. "
                "Check that the RHS was built with DE.expr() or DE.<depVar>()."
            )
        return sp.simplify(solutions[0])

    def generate_amp_factor(self) -> "DifferentialEquation":
        """Compute and cache the amplification factor G = e^{αΔt}."""
        G_rhs = self._compute_amp_factor_rhs()
        self._amp_factor   = sp.Eq(sp.exp(sp.Symbol("alpha") * self.t["variation"]), G_rhs)
        self._amp_exponent = sp.ratsimp(sp.log(G_rhs) / self.t["variation"])
        return self

    # ── modified equation ────────────────────────────────────────────────────

    def _infer_max_order(self, alpha: sp.Expr) -> int:
        deltas     = [self.vars[v]["variation"] for v in self._indep_names]
        max_orders: list[int] = []
        for delta in deltas:
            powers = {
                abs(int(exp))
                for base, exp in (
                    t.as_base_exp() for t in sp.preorder_traversal(alpha)
                )
                if base == delta and exp.is_integer
            }
            max_orders.append(max(powers, default=0))

        ranges    = [range(-m, m + 1) for m in max_orders]
        cross_max = 0
        for combo in product(*ranges):
            if sum(abs(c) for c in combo) == 0:
                continue
            term = sp.Mul(*[d**p for d, p in zip(deltas, combo)])
            if alpha.has(term):
                cross_max = max(cross_max, sum(abs(c) for c in combo))

        return max(max(max_orders, default=0), cross_max)

    def generate_modified_equation(self, nterms: int = 2) -> "DifferentialEquation":
        """
        Compute the modified equation up to *nterms* terms beyond leading order.

        Parameters
        ----------
        nterms: Number of terms (default 2). Must be ≥ 1.

        Returns *self* for chaining.
        """
        if nterms < 1:
            raise ValueError("nterms must be ≥ 1")

        if self._amp_factor is None:
            self.generate_amp_factor()

        alpha       = self._amp_exponent
        total_order = self._infer_max_order(alpha) + nterms
        k_zero      = {self.vars[v]["waveNum"]: sp.Integer(0) for v in self._indep_names}

        coefs: dict[str, sp.Expr] = {}
        derivs: dict[str, sp.Expr] = {}

        for combo in product(range(total_order), repeat=len(self._indep_names)):
            if not (0 < sum(combo) < total_order):
                continue
            N   = sum(combo)
            wrt = []
            for var_name, p in zip(self._indep_names, combo):
                wrt += [self.vars[var_name]["waveNum"], p]

            deriv_alpha = sp.diff(alpha, *wrt).subs(k_zero)
            fac         = _factorial_product(combo)
            coef        = sp.simplify(deriv_alpha / (fac * sp.I**N))
            coef        = sp.nsimplify(
                coef.evalf(), tolerance=self.__simplification_tolerance, rational=False
            )
            coef = sp.simplify(coef)

            if coef == 0:
                continue

            key        = "".join(str(p) for p in combo)
            coefs[key] = coef

            wrt_sym = []
            for var_name, p in zip(self._indep_names, combo):
                wrt_sym += [self.vars[var_name]["sym"], p]
            derivs[key] = sp.Derivative(self.dependentVar, *wrt_sym)

        self._me_coefs  = coefs
        self._me_derivs = derivs

        me_lhs          = sp.Derivative(self.dependentVar, self.t["sym"], 1)
        me_rhs          = sum(coefs[k] * derivs[k] for k in coefs) if coefs else sp.Integer(0)
        self._modified_eq = sp.Eq(me_lhs, me_rhs)
        self._latex_me    = self._build_latex()
        return self

    # ── latex builder ────────────────────────────────────────────────────────

    def _build_latex(self) -> str:
        lhs_str = sp.latex(sp.Derivative(self.dependentVar, self.t["sym"]))

        order_groups: dict[int, str] = {}
        for key in sorted(self._me_coefs, key=lambda k: sum(int(c) for c in k)):
            N     = sum(int(c) for c in key)
            coef  = self._me_coefs[key]
            deriv = self._me_derivs[key]

            coef_str  = sp.latex(coef)
            deriv_str = sp.latex(deriv)

            if coef.is_Add:
                term_str = rf"\left({coef_str}\right) {deriv_str}"
            else:
                term_str = rf"{coef_str} {deriv_str}"

            if N in order_groups:
                order_groups[N] += (" " if term_str.startswith("-") else " + ") + term_str
            else:
                order_groups[N] = term_str

        rhs_str = " ".join(
            (" " if s.startswith("-") else " + ") + s if idx > 0 else s
            for idx, s in enumerate(order_groups[kk] for kk in sorted(order_groups))
        )
        return lhs_str + " = " + rhs_str

    # ── public accessors ─────────────────────────────────────────────────────

    def symbolic_modified_equation(self) -> sp.Eq:
        """Return the symbolic modified equation (sympy Eq object)."""
        if self._modified_eq is None:
            raise RuntimeError(
                "Modified equation not yet generated. "
                "Call generate_modified_equation() first."
            )
        return self._modified_eq

    def symbolic_amp_factor(self) -> sp.Eq:
        """Return the symbolic amplification factor (sympy Eq object)."""
        if self._amp_factor is None:
            raise RuntimeError(
                "Amplification factor not yet generated. "
                "Call generate_amp_factor() or generate_modified_equation() first."
            )
        return self._amp_factor

    def latex_modified_equation(self) -> str:
        """Return the LaTeX string of the modified equation."""
        if self._modified_eq is None:
            raise RuntimeError(
                "Modified equation not yet generated. "
                "Call generate_modified_equation() first."
            )
        return self._latex_me

    def latex_amp_factor(self) -> str:
        """Return the LaTeX string of the amplification factor."""
        if self._amp_factor is None:
            raise RuntimeError(
                "Amplification factor not yet generated. "
                "Call generate_amp_factor() or generate_modified_equation() first."
            )
        return sp.latex(self._amp_factor)

    def me_coefficients(self) -> dict[str, sp.Expr]:
        """
        Return a dict mapping multi-index strings to symbolic coefficient expressions.

        Keys: ``'1'``, ``'2'``, ``'11'``, ``'20'``, etc.
        """
        if not self._me_coefs:
            raise RuntimeError("Call generate_modified_equation() first.")
        return dict(self._me_coefs)

    def independentVars(self) -> list[str]:
        """Return the list of independent variable name strings."""
        return list(self._indep_names)

    # ── tolerance property ───────────────────────────────────────────────────

    @property
    def simplification_tolerance(self) -> float:
        return self.__simplification_tolerance

    @simplification_tolerance.setter
    def simplification_tolerance(self, val: float) -> None:
        if val <= 0:
            raise ValueError("tolerance must be positive")
        self.__simplification_tolerance = val

    # ── display ──────────────────────────────────────────────────────────────

    def display_modified_equation(self) -> None:
        """Render the modified equation — LaTeX in Jupyter, symbolic in terminal."""
        if self._modified_eq is None:
            raise RuntimeError("Call generate_modified_equation() first.")
        if self._is_jupyter:
            from IPython.display import display, Math  # type: ignore
            display(Math(self._latex_me))
        else:
            sp.pprint(self._modified_eq)

    def display_amp_factor(self) -> None:
        """Render the amplification factor — LaTeX in Jupyter, symbolic in terminal."""
        if self._amp_factor is None:
            raise RuntimeError(
                "Call generate_amp_factor() or generate_modified_equation() first."
            )
        if self._is_jupyter:
            from IPython.display import display, Math  # type: ignore
            display(Math(sp.latex(self._amp_factor)))
        else:
            sp.pprint(self._amp_factor)

    def __repr__(self) -> str:
        status = "ready" if self._modified_eq is not None else "no ME yet"
        return (
            f"DifferentialEquation(dependentVarName='{self._dep_name}', "
            f"independentVarsNames={self._indep_names}, {status})"
        )
