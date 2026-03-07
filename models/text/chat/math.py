import sympy as sp

# Define the symbol
x = sp.symbols('x')

# Define the equation (x^2 + 2x - 8 = 0)
equation = sp.Eq(x**2 + 6*x - 6, 0)

# Solve for x
solution = sp.solve(equation, x)
print(f"Solutions: {solution}")