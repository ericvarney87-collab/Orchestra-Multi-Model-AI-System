"""
Math Expert Handler - Routes and processes mathematical computation queries
Integrates MathEngine with Orchestra's expert system
"""

import re
from math_engine import MathEngine

class MathExpertHandler:
    """Handles mathematical computation requests for Math_Expert domain"""
    
    def __init__(self):
        self.engine = MathEngine()
        
        # Operation patterns for auto-detection
        self.patterns = {
            'limit': r'lim(?:it)?.*?(?:as|→|->)',
            'epsilon_delta': r'epsilon.*?delta|ε.*?δ|prove.*?limit',
            'derivative': r'd/d|derivative|differentiate|d²/d|∂/∂',
            'integral': r'∫|integrate|integral',
            'ode': r"y'|y''|y'''|differential equation|ode",
            'solve': r'solve|find [xyz]|roots of',
            'matrix': r'matrix|determinant|eigenvalue|eigenvector',
            'series': r'taylor|maclaurin|series expansion',
            'summation': r'sum.*?from|∑|summation',
            'simplify': r'simplify|factor|expand',
            'gradient': r'gradient|∇|nabla'
        }
    
    def process_math_query(self, query):
        """
        Process a Math: query and return structured computation results
        
        Args:
            query: String after "Math:" prefix
            
        Returns:
            dict with computation results and formatted output
        """
        # Detect operation type
        operation = self._detect_operation(query)
        
        # Route to appropriate computation
        if operation == 'epsilon_delta':
            return self._handle_epsilon_delta(query)
        elif operation == 'limit':
            return self._handle_limit(query)
        elif operation == 'derivative':
            return self._handle_derivative(query)
        elif operation == 'integral':
            return self._handle_integral(query)
        elif operation == 'ode':
            return self._handle_ode(query)
        elif operation == 'solve':
            return self._handle_solve(query)
        elif operation == 'matrix':
            return self._handle_matrix(query)
        elif operation == 'series':
            return self._handle_series(query)
        elif operation == 'summation':
            return self._handle_summation(query)
        elif operation == 'gradient':
            return self._handle_gradient(query)
        elif operation == 'simplify':
            return self._handle_simplify(query)
        else:
            # General math evaluation
            return self._handle_general(query)
    
    def _detect_operation(self, query):
        """Detect which mathematical operation is being requested"""
        query_lower = query.lower()
        
        # Check patterns in priority order
        for op, pattern in self.patterns.items():
            if re.search(pattern, query_lower):
                return op
        
        return 'general'
    
    # =====================================================================
    # LIMIT HANDLERS
    # =====================================================================
    
    def _handle_limit(self, query):
        """
        Parse and handle limit computation
        Examples:
        - "lim (sin(x)/x) as x approaches 0"
        - "Lim of (x - y)/(y^2 - x^2) as (x,y) approaches (1,1)"
        - "Lim as x tends to infinity x!/x^x"
        """
        try:
            # Remove common prefix words
            query_clean = re.sub(r'^(solve|find|compute|calculate)\s+', '', query, flags=re.IGNORECASE)
            
            expression = None
            
            # Format 1: "Lim of <expression> as ..."
            expr_match = re.search(r'(?:lim(?:it)?|Lim)\s+(?:of\s+)?(.+?)\s+as\s+', query_clean, re.IGNORECASE)
            
            if expr_match:
                expression = expr_match.group(1).strip()
            else:
                # Format 2: "Lim as ... of <expression>"
                expr_match2 = re.search(r'(?:lim(?:it)?|Lim)\s+as\s+.+?\s+of\s+(.+)', query_clean, re.IGNORECASE)
                if expr_match2:
                    expression = expr_match2.group(1).strip()
                else:
                    # Format 3: "Lim as x tends to infinity <expression>" (no "of")
                    # Extract everything after the limit point
                    expr_match3 = re.search(r'(?:lim(?:it)?|Lim)\s+as\s+\w+\s+(?:approaches|tends to|→|->|goes to)\s+(?:\d+(?:\.\d+)?|∞|oo|infinity|inf)\s+(.+)', query_clean, re.IGNORECASE)
                    if expr_match3:
                        expression = expr_match3.group(1).strip()
            
            if not expression:
                return {"error": "Could not parse limit expression"}
            
            # Convert factorial notation: x! -> factorial(x)
            expression = re.sub(r'(\w+)!', r'factorial(\1)', expression)
            
            # Remove outer parentheses if they wrap the entire expression
            if expression.startswith('(') and expression.endswith(')'):
                depth = 0
                for i, ch in enumerate(expression):
                    if ch == '(':
                        depth += 1
                    elif ch == ')':
                        depth -= 1
                    if depth == 0 and i < len(expression) - 1:
                        break
                if depth == 0 and i == len(expression) - 1:
                    expression = expression[1:-1]
            
            print(f"DEBUG: Parsed expression: '{expression}'")
            
            # Check if it's multivariable limit
            multi_var_match = re.search(r'\(([^)]+)\)\s*(?:approaches|tends to|→|->|goes to)\s*\(([^)]+)\)', query_clean, re.IGNORECASE)
            
            if multi_var_match:
                var_str = multi_var_match.group(1)
                point_str = multi_var_match.group(2)
                
                vars_list = [v.strip() for v in var_str.split(',')]
                points_list = [float(p.strip()) for p in point_str.split(',')]
                
                print(f"DEBUG: Multivariable - vars: {vars_list}, points: {points_list}")
                
                if len(vars_list) != len(points_list):
                    return {"error": "Number of variables doesn't match number of points"}
                
                var_tuple = tuple(vars_list)
                point_tuple = tuple(points_list)
                
                result = self.engine.compute_limit(expression, var_tuple, point_tuple)
                
                if result.get("error"):
                    return result
                
                return {
                    "operation": "limit",
                    "result": result,
                    "formatted_output": self._format_limit_result(result, expression, var_tuple, point_tuple, None),
                    "latex_available": True
                }
            
            # Single variable limit
            var_match = re.search(r'(?:as|when)\s+([xyztn])\s*(?:approaches|tends to|→|->|goes to)\s*(-?\d+(?:\.\d+)?|∞|oo|infinity|inf)', query_clean, re.IGNORECASE)
            if not var_match:
                return {"error": "Could not parse variable and point"}
            
            var = var_match.group(1)
            point_str = var_match.group(2)
            
            # Handle infinity
            if point_str.lower() in ['∞', 'oo', 'infinity', 'inf']:
                point = 'oo'
            else:
                point = float(point_str)
            
            # Check for direction
            direction = None
            if re.search(r'from\s+(?:the\s+)?right|[+]', query_clean):
                direction = '+'
            elif re.search(r'from\s+(?:the\s+)?left|[-](?!>)', query_clean):
                direction = '-'
            
            # Compute limit
            result = self.engine.compute_limit(expression, var, point, direction)
            
            if result.get("error"):
                return result
            
            return {
                "operation": "limit",
                "result": result,
                "formatted_output": self._format_limit_result(result, expression, var, point, direction),
                "latex_available": True
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Limit parsing error: {str(e)}"}
    
    # =====================================================================
    # DERIVATIVE HANDLERS
    # =====================================================================
    
    def _handle_derivative(self, query):
        """
        Parse and handle derivative computation
        Examples:
        - "derivative of x^2 + 3x with respect to x"
        - "d/dx (sin(x) * cos(x))"
        - "second derivative of e^x"
        """
        try:
            # Extract order
            order = 1
            if 'second' in query.lower() or 'd²/d' in query or 'd^2/d' in query:
                order = 2
            elif 'third' in query.lower() or 'd³/d' in query or 'd^3/d' in query:
                order = 3
            
            # Check if it's a partial derivative
            is_partial = '∂' in query or 'partial' in query.lower()
            
            # Extract expression and variable
            # Pattern 1: "derivative of <expr> with respect to <var>"
            match1 = re.search(r'derivative\s+of\s+(.+?)\s+with respect to\s+([xyz])', query, re.IGNORECASE)
            # Pattern 2: "d/dx (<expr>)" or "d/dx <expr>"
            match2 = re.search(r'd/d([xyz])\s*\(([^)]+)\)', query)
            match3 = re.search(r'd/d([xyz])\s+(.+?)(?:\s+|$)', query)
            
            if match1:
                expression = match1.group(1).strip()
                var = match1.group(2)
            elif match2:
                var = match2.group(1)
                expression = match2.group(2).strip()
            elif match3:
                var = match3.group(1)
                expression = match3.group(2).strip()
            else:
                return {"error": "Could not parse derivative expression"}
            
            # Compute derivative
            if is_partial:
                result = self.engine.partial_derivative(expression, var, order)
            else:
                result = self.engine.derivative(expression, var, order)
            
            if result.get("error"):
                return result
            
            return {
                "operation": "derivative",
                "result": result,
                "formatted_output": self._format_derivative_result(result, expression, var, order, is_partial),
                "latex_available": True
            }
            
        except Exception as e:
            return {"error": f"Derivative parsing error: {str(e)}"}
    
    def _handle_gradient(self, query):
        """Handle gradient computation"""
        try:
            # Extract expression
            expr_match = re.search(r'(?:gradient|∇|nabla)\s+(?:of\s+)?(.+?)(?:\s+at|\s+in terms of|$)', query, re.IGNORECASE)
            if not expr_match:
                return {"error": "Could not parse gradient expression"}
            
            expression = expr_match.group(1).strip()
            
            # Determine variables (default to x, y, z if in expression)
            vars_in_expr = []
            for v in ['x', 'y', 'z']:
                if v in expression:
                    vars_in_expr.append(v)
            
            if not vars_in_expr:
                return {"error": "No variables detected in expression"}
            
            result = self.engine.gradient(expression, vars_in_expr)
            
            if result.get("error"):
                return result
            
            return {
                "operation": "gradient",
                "result": result,
                "formatted_output": self._format_gradient_result(result, expression),
                "latex_available": True
            }
            
        except Exception as e:
            return {"error": f"Gradient parsing error: {str(e)}"}
    
    # =====================================================================
    # INTEGRAL HANDLERS
    # =====================================================================
    
    def _handle_integral(self, query):
        """
        Parse and handle integral computation
        Examples:
        - "integrate x^2 from 0 to 1"
        - "∫ sin(x) dx"
        - "definite integral of e^x from 1 to 2"
        """
        try:
            # Determine if definite or indefinite
            has_bounds = bool(re.search(r'from\s+(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)', query, re.IGNORECASE))
            
            # Extract expression
            expr_match = re.search(r'(?:integrate|integral of|∫)\s*(.+?)(?:\s+d[xyz]|\s+from|\s+with respect to|$)', query, re.IGNORECASE)
            if not expr_match:
                return {"error": "Could not parse integral expression"}
            
            expression = expr_match.group(1).strip()
            
            # Extract variable
            var_match = re.search(r'd([xyz])|with respect to\s+([xyz])', query)
            if var_match:
                var = var_match.group(1) or var_match.group(2)
            else:
                # Auto-detect
                for v in ['x', 'y', 'z']:
                    if v in expression:
                        var = v
                        break
                else:
                    var = 'x'
            
            # Extract bounds if definite
            lower, upper = None, None
            if has_bounds:
                bounds_match = re.search(r'from\s+(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)', query, re.IGNORECASE)
                if bounds_match:
                    lower = float(bounds_match.group(1))
                    upper = float(bounds_match.group(2))
            
            # Compute integral
            result = self.engine.integral(expression, var, lower, upper)
            
            if result.get("error"):
                return result
            
            return {
                "operation": "integral",
                "result": result,
                "formatted_output": self._format_integral_result(result, expression, var, lower, upper),
                "latex_available": True
            }
            
        except Exception as e:
            return {"error": f"Integral parsing error: {str(e)}"}
    
    # =====================================================================
    # EQUATION SOLVING
    # =====================================================================
    
    def _handle_solve(self, query):
        """
        Parse and handle equation solving
        Examples:
        - "solve x^2 + 5x + 6 = 0"
        - "find x in 2x + 3 = 7"
        """
        try:
            # Extract equation
            eq_match = re.search(r'(?:solve|find [xyz])(.*?)(?:for [xyz]|$)', query, re.IGNORECASE)
            if not eq_match:
                return {"error": "Could not parse equation"}
            
            equation = eq_match.group(1).strip()
            # Clean up extra words
            equation = re.sub(r'\s+(?:in|where|such that)\s+', ' ', equation)
            
            # Extract variable if specified
            var_match = re.search(r'for\s+([xyz])', query)
            var = var_match.group(1) if var_match else None
            
            # Solve
            result = self.engine.solve_equation(equation, var)
            
            if result.get("error"):
                return result
            
            return {
                "operation": "solve",
                "result": result,
                "formatted_output": self._format_solve_result(result, equation),
                "latex_available": True
            }
            
        except Exception as e:
            return {"error": f"Equation solving error: {str(e)}"}
    
    # =====================================================================
    # OTHER OPERATIONS
    # =====================================================================
    
    def _handle_ode(self, query):
        """Handle differential equation solving"""
        try:
            # Extract the ODE
            ode_match = re.search(r'(?:solve|find)\s+(.+)', query, re.IGNORECASE)
            if not ode_match:
                return {"error": "Could not parse ODE"}
            
            ode_str = ode_match.group(1).strip()
            
            result = self.engine.solve_ode(ode_str)
            
            if result.get("error"):
                return result
            
            return {
                "operation": "ode",
                "result": result,
                "formatted_output": self._format_ode_result(result),
                "latex_available": True
            }
            
        except Exception as e:
            return {"error": f"ODE solving error: {str(e)}"}
    
    def _handle_matrix(self, query):
        """Handle matrix operations"""
        # This would need matrix parsing - placeholder for now
        return {"error": "Matrix operations require explicit matrix input"}
    
    def _handle_series(self, query):
        """Handle Taylor/Maclaurin series"""
        try:
            # Extract expression, point, and order
            expr_match = re.search(r'(?:taylor|maclaurin)\s+(?:series\s+)?(?:of\s+)?(.+?)(?:\s+at|\s+centered|\s+order)', query, re.IGNORECASE)
            if not expr_match:
                return {"error": "Could not parse series expression"}
            
            expression = expr_match.group(1).strip()
            
            # Extract point
            point_match = re.search(r'(?:at|centered at|around)\s+([xyz])\s*=\s*(-?\d+)', query, re.IGNORECASE)
            if point_match:
                var = point_match.group(1)
                point = int(point_match.group(2))
            else:
                var = 'x'
                point = 0  # Maclaurin default
            
            # Extract order
            order_match = re.search(r'order\s+(\d+)', query, re.IGNORECASE)
            order = int(order_match.group(1)) if order_match else 5
            
            result = self.engine.taylor_series(expression, var, point, order)
            
            if result.get("error"):
                return result
            
            return {
                "operation": "series",
                "result": result,
                "formatted_output": self._format_series_result(result),
                "latex_available": True
            }
            
        except Exception as e:
            return {"error": f"Series computation error: {str(e)}"}
    
    def _handle_summation(self, query):
        """Handle summation computation"""
        # Placeholder - would need summation parsing
        return {"error": "Summation requires more specific syntax"}
    
    def _handle_simplify(self, query):
        """Handle expression simplification"""
        try:
            # Determine method
            method = 'simplify'
            if 'expand' in query.lower():
                method = 'expand'
            elif 'factor' in query.lower():
                method = 'factor'
            
            # Extract expression
            expr_match = re.search(r'(?:simplify|expand|factor)\s+(.+)', query, re.IGNORECASE)
            if not expr_match:
                return {"error": "Could not parse expression"}
            
            expression = expr_match.group(1).strip()
            
            result = self.engine.simplify_expression(expression, method)
            
            if result.get("error"):
                return result
            
            return {
                "operation": "simplify",
                "result": result,
                "formatted_output": self._format_simplify_result(result),
                "latex_available": True
            }
            
        except Exception as e:
            return {"error": f"Simplification error: {str(e)}"}
    
    def _handle_general(self, query):
        """Handle general mathematical query"""
        return {
            "operation": "general",
            "message": "Could not auto-detect operation. Please use explicit format like: 'solve ...', 'derivative of ...', 'limit of ...'"
        }
    
    # =====================================================================
    # FORMATTING FUNCTIONS
    # =====================================================================
    
    def _format_epsilon_delta_proof(self, proof):
        """Format epsilon-delta proof for display"""
        output = [f"EPSILON-DELTA PROOF"]
        output.append(f"\n{proof['statement']}")
        output.append(f"\n✓ Computed Limit: {proof['computed_limit']}")
        output.append(f"✓ Verification: {'PASSED' if proof['verified'] else 'FAILED'}")
        
        output.append(f"\nProof Structure:")
        for key, value in proof['proof_structure'].items():
            output.append(f"  {key}: {value}")
        
        output.append(f"\nLaTeX: {proof['latex_statement']}")
        
        return "\n".join(output)
    
    def _format_limit_result(self, result, expr, var, point, direction):
        """Format limit computation result"""
        output = [f"LIMIT COMPUTATION"]
        
        # Handle multivariable
        if isinstance(var, tuple):
            var_str = str(var)
            point_str = str(point)
            output.append(f"\nlim ({expr}) as {var_str} → {point_str}")
        else:
            dir_text = f" from {'right' if direction == '+' else 'left'}" if direction else ""
            output.append(f"\nlim ({expr}) as {var} → {point}{dir_text}")
        
        output.append(f"\n✓ Result: {result['result']}")
        if result.get('numerical_value'):
            output.append(f"  Numerical: {result['numerical_value']}")
        output.append(f"\nLaTeX: {result.get('latex_full', result['latex'])}")
        
        return "\n".join(output)
    
    def _format_derivative_result(self, result, expr, var, order, is_partial):
        """Format derivative result"""
        output = [f"DERIVATIVE COMPUTATION"]
        deriv_type = "Partial" if is_partial else "Derivative"
        order_text = f"{order}" if order > 1 else ""
        output.append(f"\n{deriv_type} (order {order}) of {expr} with respect to {var}")
        output.append(f"\n✓ Result: {result['simplified']}")
        output.append(f"\nLaTeX: {result['latex_notation']} = {result['latex_result']}")
        
        return "\n".join(output)
    
    def _format_gradient_result(self, result, expr):
        """Format gradient result"""
        output = [f"GRADIENT COMPUTATION"]
        output.append(f"\n∇({expr})")
        output.append(f"\n✓ Result: {result['latex']}")
        
        return "\n".join(output)
    
    def _format_integral_result(self, result, expr, var, lower, upper):
        """Format integral result"""
        output = [f"INTEGRAL COMPUTATION"]
        if result['is_definite']:
            output.append(f"\nDefinite integral of {expr} from {lower} to {upper}")
        else:
            output.append(f"\nIndefinite integral of {expr} with respect to {var}")
        
        output.append(f"\n✓ Result: {result['simplified']}")
        if result.get('numerical'):
            output.append(f"  Numerical value: {result['numerical']}")
        output.append(f"\nLaTeX: {result['latex_result']}")
        
        return "\n".join(output)
    
    def _format_solve_result(self, result, equation):
        """Format equation solving result"""
        output = [f"EQUATION SOLVING"]
        output.append(f"\nSolve: {equation}")
        output.append(f"\n✓ Solutions ({result['count']}):")
        for i, sol_latex in enumerate(result['latex_formatted']):
            output.append(f"  {i+1}. {sol_latex}")
            if result['numerical'][i] is not None:
                output.append(f"      ≈ {result['numerical'][i]}")
        
        return "\n".join(output)
    
    def _format_ode_result(self, result):
        """Format ODE solving result"""
        output = [f"DIFFERENTIAL EQUATION SOLVING"]
        output.append(f"\n✓ Solution: {result['latex']}")
        
        return "\n".join(output)
    
    def _format_series_result(self, result):
        """Format series expansion result"""
        output = [f"TAYLOR SERIES EXPANSION"]
        output.append(f"\nOrder {result['order']} at x = {result['point']}")
        output.append(f"\n✓ Result: {result['latex']}")
        
        return "\n".join(output)
    
    def _format_simplify_result(self, result):
        """Format simplification result"""
        output = [f"EXPRESSION SIMPLIFICATION ({result['method'].upper()})"]
        output.append(f"\nOriginal: {result['latex_original']}")
        output.append(f"Result:   {result['latex_result']}")
        
        return "\n".join(output)
