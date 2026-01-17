"""
Math Engine - Comprehensive symbolic mathematics and computation
Handles limits, derivatives, integrals, differential equations, linear algebra,
series, equation solving, and proof frameworks
"""

import sympy as sp
from sympy import (
    symbols, Symbol, Function,
    limit, diff, integrate, solve, solveset, dsolve,
    simplify, expand, factor, cancel, apart, together,
    latex, pretty,
    Matrix, det,
    series, summation, product,
    sqrt, exp, log, sin, cos, tan, sinh, cosh, tanh,
    I, pi, E, oo, Rational,
    Eq, Derivative, Integral
)
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import numpy as np
import re

class MathEngine:
    """Interface for symbolic mathematics and rigorous computation"""
    
    def __init__(self):
        self.available = self._check_available()
        # Pre-define common symbols
        self.x, self.y, self.z = symbols('x y z', real=True)
        self.t, self.n, self.k = symbols('t n k', real=True)
        self.a, self.b, self.c = symbols('a b c', real=True)
        
        # Parsing transformations
        self.transformations = (standard_transformations + 
                               (implicit_multiplication_application,))
    
    def _check_available(self):
        """Check if SymPy is installed"""
        try:
            import sympy
            import numpy
            return True
        except ImportError:
            return False
    
    def _parse_expression(self, expr_str):
        """Parse string to SymPy expression"""
        try:
            # Convert ^ to ** for exponents
            expr_str = expr_str.replace('^', '**')
            
            from sympy import factorial
            
            local_dict = {
                'x': self.x, 'y': self.y, 'z': self.z,
                't': self.t, 'n': self.n, 'k': self.k,
                'a': self.a, 'b': self.b, 'c': self.c,
                'pi': pi, 'e': E, 'I': I, 'oo': oo,
                'sin': sin, 'cos': cos, 'tan': tan,
                'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
                'sqrt': sqrt, 'exp': exp, 'log': log, 'ln': log,
                'factorial': factorial  # ADD THIS LINE
            }
            return parse_expr(expr_str, local_dict=local_dict, 
                            transformations=self.transformations)
        except Exception as e:
            return None
    
    # =====================================================================
    # LIMITS
    # =====================================================================
    
    def compute_limit(self, expression_str, var, point, direction=None):
        """
        Compute limit of expression as var approaches point
        
        Args:
            expression_str: String like "sin(x)/x" or "2*x + 3*y"
            var: Variable(s) - single string 'x' or tuple ('x','y')
            point: Point - single value or tuple (1,3)
            direction: '+', '-', or None for bidirectional
            
        Returns:
            dict with result, latex, steps, and verification
        """
        try:
            expr = self._parse_expression(expression_str)
            if expr is None:
                return {"error": "Could not parse expression"}
            
            # Check if expression involves factorial (special handling needed)
            has_factorial = 'factorial' in str(expr)
            
            # Handle multivariable limits
            if isinstance(var, tuple) and isinstance(point, tuple):
                result = expr
                steps = []
                latex_steps = []
                
                for v, p in zip(var, point):
                    var_sym = symbols(v, real=True)
                    prev_result = result
                    result = limit(result, var_sym, p)
                    steps.append(f"lim({prev_result}) as {v}→{p} = {result}")
                    latex_steps.append(f"\\lim_{{{v} \\to {p}}} {latex(prev_result)} = {latex(result)}")
                
                return {
                    "result": result,
                    "latex": latex(result),
                    "expression": expr,
                    "steps": steps,
                    "latex_steps": latex_steps,
                    "numerical_value": float(result.evalf()) if result.is_number else None,
                    "is_finite": result.is_finite
                }
            
            # Single variable limit
            var_sym = symbols(var, real=True)
            
            if direction:
                result = limit(expr, var_sym, point, dir=direction)
            else:
                result = limit(expr, var_sym, point)
            
            response = {
                "result": result,
                "latex": latex(result),
                "latex_full": f"\\lim_{{{var} \\to {point}}} {latex(expr)} = {latex(result)}",
                "expression": expr,
                "steps": [f"lim({expr}) as {var}→{point} = {result}"],
                "numerical_value": float(result.evalf()) if result.is_number else None,
                "is_finite": result.is_finite,
                "direction": direction
            }
            
            # Add asymptotic analysis for factorial limits at infinity
            if has_factorial and point == 'oo':
                response["asymptotic_note"] = self._analyze_factorial_limit(expr, var_sym)
            
            return response
            
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_factorial_limit(self, expr, var):
        """Provide asymptotic analysis for limits involving factorials"""
        try:
            expr_str = str(expr)
            
            # Common patterns
            if 'factorial' in expr_str and '**' in expr_str:
                note = """
ASYMPTOTIC ANALYSIS (Stirling's Approximation):
n! ≈ √(2πn) · (n/e)^n

Key growth rates as n → ∞:
- n^n grows FASTER than n! (denominator dominates)
- n! grows FASTER than e^n
- n! grows FASTER than any polynomial

For n!/n^n specifically:
n!/n^n ≈ √(2πn) · (n/e)^n / n^n
       = √(2πn) · (1/e)^n
       → 0 as n → ∞

CONCLUSION: The limit is 0 because the DENOMINATOR (n^n) grows exponentially 
faster than the numerator (n!). Each is vastly larger than the previous, but 
n^n outpaces n! due to the additional factor of e^n in the denominator.
                """
                return note.strip()
            
            return None
        except:
            return None
    
    # =====================================================================
    # DERIVATIVES
    # =====================================================================
    
    def derivative(self, expression_str, var, order=1):
        """
        Compute derivative
        
        Args:
            expression_str: Expression to differentiate
            var: Variable to differentiate with respect to
            order: Order of derivative (default 1)
            
        Returns:
            dict with result, steps, and latex
        """
        try:
            expr = self._parse_expression(expression_str)
            if expr is None:
                return {"error": "Could not parse expression"}
            
            var_sym = symbols(var, real=True)
            
            # Compute derivatives step by step
            steps = [expr]
            for i in range(order):
                steps.append(diff(steps[-1], var_sym))
            
            result = steps[-1]
            simplified = simplify(result)
            
            return {
                "result": result,
                "simplified": simplified,
                "latex": latex(simplified),
                "latex_notation": f"\\frac{{d^{order}}}{{d{var}^{order}}} {latex(expr)}" if order > 1 else f"\\frac{{d}}{{d{var}}} {latex(expr)}",
                "latex_result": f"{latex(simplified)}",
                "steps": [latex(s) for s in steps],
                "order": order
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def partial_derivative(self, expression_str, var, order=1):
        """Compute partial derivative (same as derivative but for clarity)"""
        result = self.derivative(expression_str, var, order)
        if not result.get("error"):
            result["latex_notation"] = f"\\frac{{\\partial^{order}}}{{\\partial {var}^{order}}} {latex(self._parse_expression(expression_str))}" if order > 1 else f"\\frac{{\\partial}}{{\\partial {var}}} {latex(self._parse_expression(expression_str))}"
        return result
    
    def gradient(self, expression_str, vars):
        """
        Compute gradient vector
        
        Args:
            expression_str: Scalar function
            vars: List of variables like ['x', 'y', 'z']
            
        Returns:
            Gradient vector
        """
        try:
            expr = self._parse_expression(expression_str)
            if expr is None:
                return {"error": "Could not parse expression"}
            
            grad_components = []
            latex_components = []
            
            for v in vars:
                var_sym = symbols(v, real=True)
                partial = diff(expr, var_sym)
                grad_components.append(partial)
                latex_components.append(latex(partial))
            
            return {
                "gradient": grad_components,
                "latex": f"\\nabla f = ({', '.join(latex_components)})",
                "components": {v: c for v, c in zip(vars, grad_components)},
                "magnitude": sqrt(sum(c**2 for c in grad_components))
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    # =====================================================================
    # INTEGRALS
    # =====================================================================
    
    def integral(self, expression_str, var, lower=None, upper=None):
        """
        Compute integral (definite or indefinite)
        
        Args:
            expression_str: Integrand
            var: Variable of integration
            lower: Lower bound (None for indefinite)
            upper: Upper bound (None for indefinite)
            
        Returns:
            dict with result, latex, and numerical value if applicable
        """
        try:
            expr = self._parse_expression(expression_str)
            if expr is None:
                return {"error": "Could not parse expression"}
            
            var_sym = symbols(var, real=True)
            
            if lower is not None and upper is not None:
                # Definite integral
                result = integrate(expr, (var_sym, lower, upper))
                integral_notation = f"\\int_{{{lower}}}^{{{upper}}} {latex(expr)} \\, d{var}"
                is_definite = True
            else:
                # Indefinite integral
                result = integrate(expr, var_sym)
                integral_notation = f"\\int {latex(expr)} \\, d{var}"
                is_definite = False
            
            simplified = simplify(result)
            
            return {
                "result": result,
                "simplified": simplified,
                "latex": latex(simplified),
                "latex_notation": integral_notation,
                "latex_result": f"{integral_notation} = {latex(simplified)}",
                "numerical": float(result.evalf()) if result.is_number else None,
                "is_definite": is_definite,
                "bounds": (lower, upper) if is_definite else None
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def multiple_integral(self, expression_str, vars_with_bounds):
        """
        Compute multiple integral
        
        Args:
            expression_str: Integrand
            vars_with_bounds: List of tuples like [('x', 0, 1), ('y', 0, 2)]
            
        Returns:
            Result of iterated integration
        """
        try:
            expr = self._parse_expression(expression_str)
            if expr is None:
                return {"error": "Could not parse expression"}
            
            # Build integration bounds in reverse order (innermost first)
            bounds = []
            for var, lower, upper in reversed(vars_with_bounds):
                var_sym = symbols(var, real=True)
                bounds.append((var_sym, lower, upper))
            
            result = integrate(expr, *bounds)
            
            # Build LaTeX notation
            integral_signs = "".join([f"\\int_{{{l}}}^{{{u}}}" for _, l, u in reversed(bounds)])
            dvars = " ".join([f"d{v}" for v, _, _ in reversed(vars_with_bounds)])
            latex_notation = f"{integral_signs} {latex(expr)} \\, {dvars}"
            
            return {
                "result": result,
                "latex": latex(result),
                "latex_notation": latex_notation,
                "latex_result": f"{latex_notation} = {latex(result)}",
                "numerical": float(result.evalf()) if result.is_number else None
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    # =====================================================================
    # DIFFERENTIAL EQUATIONS
    # =====================================================================
    
    def solve_ode(self, equation_str, function_name='y', var='x'):
        """
        Solve ordinary differential equation
        
        Args:
            equation_str: ODE like "y'' + 2*y' + y = 0" or "Derivative(y(x), x, 2) + y(x) = 0"
            function_name: Name of unknown function (default 'y')
            var: Independent variable (default 'x')
            
        Returns:
            General solution or particular solution if initial conditions given
        """
        try:
            var_sym = symbols(var, real=True)
            func = Function(function_name)
            
            # Parse equation - handle y', y'', etc.
            eq_str = equation_str
            eq_str = eq_str.replace("y'''", "Derivative(y(x), x, 3)")
            eq_str = eq_str.replace("y''", "Derivative(y(x), x, 2)")
            eq_str = eq_str.replace("y'", "Derivative(y(x), x)")
            eq_str = eq_str.replace("y", "y(x)")
            
            # Parse the equation
            if '=' in eq_str:
                left, right = eq_str.split('=')
                left_expr = self._parse_expression(left.strip())
                right_expr = self._parse_expression(right.strip())
                equation = Eq(left_expr, right_expr)
            else:
                equation = Eq(self._parse_expression(eq_str), 0)
            
            # Solve
            solution = dsolve(equation, func(var_sym))
            
            return {
                "solution": solution,
                "latex": latex(solution),
                "equation": equation,
                "latex_equation": latex(equation),
                "general_solution": True
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    # =====================================================================
    # EQUATION SOLVING
    # =====================================================================
    
    def solve_equation(self, equation_str, var=None):
        """
        Solve equation(s) for variable(s)
        
        Args:
            equation_str: Equation like "x^2 + 5*x + 6 = 0" or "x^2 + 5*x + 6"
            var: Variable to solve for (auto-detect if None)
            
        Returns:
            dict with solutions
        """
        try:
            # Parse equation
            if '=' in equation_str:
                left, right = equation_str.split('=')
                left_expr = self._parse_expression(left.strip())
                right_expr = self._parse_expression(right.strip())
                expr = left_expr - right_expr
            else:
                expr = self._parse_expression(equation_str)
            
            if expr is None:
                return {"error": "Could not parse equation"}
            
            # Auto-detect variable if not specified
            if var is None:
                free_symbols = list(expr.free_symbols)
                if len(free_symbols) == 0:
                    return {"error": "No variables found in equation"}
                var_sym = free_symbols[0]
                var = str(var_sym)
            else:
                # Use the symbol from the expression if it exists
                free_symbols = list(expr.free_symbols)
                var_sym = None
                for sym in free_symbols:
                    if str(sym) == var:
                        var_sym = sym
                        break
                if var_sym is None:
                    var_sym = symbols(var)
            
            # Solve
            solutions = solve(expr, var_sym)
            
            # Format solutions
            formatted_solutions = []
            latex_solutions = []
            
            for sol in solutions:
                formatted_solutions.append(sol)
                latex_solutions.append(latex(sol))
            
            return {
                "solutions": formatted_solutions,
                "latex": latex_solutions,
                "latex_formatted": [f"{var} = {latex(sol)}" for sol in formatted_solutions],
                "count": len(solutions),
                "variable": var,
                "numerical": [float(sol.evalf()) if sol.is_number else None for sol in solutions]
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def solve_system(self, equations_list, vars_list):
        """
        Solve system of equations
        
        Args:
            equations_list: List of equation strings
            vars_list: List of variables to solve for
            
        Returns:
            dict with solutions
        """
        try:
            vars_sym = [symbols(v) for v in vars_list]
            equations = []
            
            for eq_str in equations_list:
                if '=' in eq_str:
                    left, right = eq_str.split('=')
                    left_expr = self._parse_expression(left.strip())
                    right_expr = self._parse_expression(right.strip())
                    equations.append(Eq(left_expr, right_expr))
                else:
                    equations.append(Eq(self._parse_expression(eq_str), 0))
            
            solutions = solve(equations, vars_sym)
            
            return {
                "solutions": solutions,
                "latex": latex(solutions),
                "variables": vars_list,
                "system_size": len(equations)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    # =====================================================================
    # LINEAR ALGEBRA
    # =====================================================================
    
    def matrix_operations(self, matrix_data, operation='info'):
        """
        Perform matrix operations
        
        Args:
            matrix_data: List of lists [[1,2],[3,4]] or Matrix object
            operation: 'info', 'inverse', 'eigenvalues', 'eigenvectors', 'determinant', 'transpose', 'rref'
            
        Returns:
            Result of operation
        """
        try:
            if isinstance(matrix_data, list):
                M = Matrix(matrix_data)
            else:
                M = matrix_data
            
            result = {}
            
            if operation == 'info' or operation == 'all':
                result['matrix'] = M
                result['latex'] = latex(M)
                result['shape'] = M.shape
                result['determinant'] = det(M) if M.is_square else None
                result['trace'] = M.trace() if M.is_square else None
                result['rank'] = M.rank()
                
            if operation == 'determinant':
                result['determinant'] = det(M)
                result['latex'] = latex(det(M))
                
            if operation == 'inverse':
                if M.is_square:
                    result['inverse'] = M.inv()
                    result['latex'] = latex(M.inv())
                else:
                    result['error'] = "Matrix must be square for inverse"
                    
            if operation == 'eigenvalues' or operation == 'eigen':
                if M.is_square:
                    eigs = M.eigenvals()
                    result['eigenvalues'] = eigs
                    result['latex'] = latex(eigs)
                else:
                    result['error'] = "Matrix must be square for eigenvalues"
                    
            if operation == 'eigenvectors' or operation == 'eigen':
                if M.is_square:
                    eigvects = M.eigenvects()
                    result['eigenvectors'] = eigvects
                    result['latex'] = latex(eigvects)
                else:
                    result['error'] = "Matrix must be square for eigenvectors"
                    
            if operation == 'transpose':
                result['transpose'] = M.T
                result['latex'] = latex(M.T)
                
            if operation == 'rref':
                rref_matrix, pivot_cols = M.rref()
                result['rref'] = rref_matrix
                result['pivot_columns'] = pivot_cols
                result['latex'] = latex(rref_matrix)
            
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    # =====================================================================
    # SERIES AND SEQUENCES
    # =====================================================================
    
    def taylor_series(self, expression_str, var, point, order):
        """
        Compute Taylor series expansion
        
        Args:
            expression_str: Function to expand
            var: Variable
            point: Point of expansion
            order: Order of expansion
            
        Returns:
            Taylor series
        """
        try:
            expr = self._parse_expression(expression_str)
            var_sym = symbols(var, real=True)
            
            taylor = series(expr, var_sym, point, n=order+1).removeO()
            
            return {
                "series": taylor,
                "latex": latex(taylor),
                "latex_full": f"{latex(expr)} \\approx {latex(taylor)} + O((x-{point})^{{{order+1}}})",
                "order": order,
                "point": point
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def sum_series(self, expression_str, var, lower, upper):
        """
        Compute summation
        
        Args:
            expression_str: Term of series
            var: Index variable
            lower: Lower bound
            upper: Upper bound (can be 'oo' for infinity)
            
        Returns:
            Sum
        """
        try:
            expr = self._parse_expression(expression_str)
            var_sym = symbols(var, integer=True)
            
            if upper == 'oo':
                upper = oo
            
            result = summation(expr, (var_sym, lower, upper))
            
            return {
                "sum": result,
                "latex": latex(result),
                "latex_notation": f"\\sum_{{{var}={lower}}}^{{{upper}}} {latex(expr)} = {latex(result)}"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    # =====================================================================
    # SIMPLIFICATION AND MANIPULATION
    # =====================================================================
    
    def simplify_expression(self, expression_str, method='simplify'):
        """
        Simplify expression using various methods
        
        Args:
            expression_str: Expression to simplify
            method: 'simplify', 'expand', 'factor', 'cancel', 'apart', 'together'
            
        Returns:
            Simplified expression
        """
        try:
            expr = self._parse_expression(expression_str)
            if expr is None:
                return {"error": "Could not parse expression"}
            
            methods = {
                'simplify': simplify,
                'expand': expand,
                'factor': factor,
                'cancel': cancel,
                'apart': apart,
                'together': together
            }
            
            if method in methods:
                result = methods[method](expr)
            else:
                result = simplify(expr)
            
            return {
                "original": expr,
                "result": result,
                "latex_original": latex(expr),
                "latex_result": latex(result),
                "method": method
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    # =====================================================================
    # UTILITY FUNCTIONS
    # =====================================================================
    
    def evaluate_at_point(self, expression_str, substitutions):
        """
        Evaluate expression at specific point
        
        Args:
            expression_str: Expression
            substitutions: Dict like {'x': 1, 'y': 2}
            
        Returns:
            Numerical result
        """
        try:
            expr = self._parse_expression(expression_str)
            if expr is None:
                return {"error": "Could not parse expression"}
            
            subs_dict = {symbols(k): v for k, v in substitutions.items()}
            result = expr.subs(subs_dict)
            
            return {
                "result": result,
                "numerical": float(result.evalf()),
                "latex": latex(result),
                "substitutions": substitutions
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def check_equality(self, expr1_str, expr2_str):
        """Check if two expressions are mathematically equal"""
        try:
            expr1 = self._parse_expression(expr1_str)
            expr2 = self._parse_expression(expr2_str)
            
            if expr1 is None or expr2 is None:
                return {"error": "Could not parse expressions"}
            
            diff = simplify(expr1 - expr2)
            equal = (diff == 0)
            
            return {
                "equal": equal,
                "difference": diff,
                "latex_diff": latex(diff),
                "expr1": latex(expr1),
                "expr2": latex(expr2)
            }
            
        except Exception as e:
            return {"error": str(e)}
