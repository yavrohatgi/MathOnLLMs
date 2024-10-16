import re

def extract_answer(solution):
    # Pattern to match either a fraction or a whole number inside \boxed{}
    pattern = r'\\boxed\{((?:\\frac\{[^}]+\}\{[^}]+\}|\d+))\}'
    match = re.search(pattern, solution)
    
    if match:
        content = match.group(1)
        if content.startswith('\\frac'):
            # Extract numerator and denominator
            frac_pattern = r'\\frac\{([^}]+)\}\{([^}]+)\}'
            frac_match = re.search(frac_pattern, content)
            if frac_match:
                numerator = frac_match.group(1)
                denominator = frac_match.group(2)
                try:
                    decimal = float(numerator) / float(denominator)
                    return f"{numerator}/{denominator}", decimal
                except ValueError:
                    return content, None
        else:
            # Whole number
            try:
                number = float(content)
                return content, number
            except ValueError:
                return content, None
    else:
        return "Answer not found", None

# Test cases
solutions = [
    'We want $\\pi R^{2}-\\pi r^{2}\\leq 5\\pi$. Dividing by $\\pi$, we have $R^{2}-r^{2}\\leq 5$. Factor the left-hand side to get $(R+r)(R-r)\\leq 5$. Substituting 10 for $R+r$ gives $10(R-r)\\leq 5 \\implies R-r \\leq 1/2$. So the maximum difference in the lengths of the radii is $\\boxed{\\frac{1}{2}}$.',
    'Completing the square, we get $f(x) = (x-4)^2 - 1$. The vertex of the graph of this equation is thus $(4, -1)$. Using the Pythagorean Theorem, it follows that the distance between $(0, 2)$ and $(4, -1)$ is $\\boxed{5}$.',
    'One hundred twenty percent of 30 is $120\\cdot30\\cdot\\frac{1}{100}=36$, and $130\\%$ of 20 is $ 130\\cdot 20\\cdot\\frac{1}{100}=26$.  The difference between 36 and 26 is $\\boxed{10}$.'
]

for i, solution in enumerate(solutions, 1):
    extracted, decimal = extract_answer(solution)
    print(f"Solution {i}:")
    print(f"Extracted answer: {extracted}")
    if decimal is not None:
        print(f"Decimal form: {decimal}")
    print()