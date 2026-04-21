def write_cnf_to_file(num_variables, clauses, file_name):
    """
    Writes CNF clauses to a file in DIMACS format.

    :param num_variables: Number of variables in the CNF formula
    :param clauses: List of clauses, where each clause is a list of integers
    :param file_name: The output file name for the CNF constraints
    """
    with open(file_name, 'w') as f:
        # Write the problem line
        f.write(f"p cnf {num_variables} {len(clauses)}\n")
        
        # Write each clause
        for clause in clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")


def latin_square_to_cnf(n):
    """
    Generates CNF clauses for constructing a Latin square of order n.

    :param n: Order of the Latin square (number of rows and columns)
    :return: A tuple of the number of variables and a list of CNF clauses
    """
    clauses = []
    num_variables = n * n * n  # Each cell can hold one of n values, so we have n^3 variables

    def varnum(row, col, val):
        """Map the 3D index (row, col, val) to a variable number for the SAT solver."""
        return row * n * n + col * n + val + 1

    # Rule 1: Each cell contains at least one number (1..n)
    for r in range(n):
        for c in range(n):
            clauses.append([varnum(r, c, v) for v in range(n)])

    # Rule 2: Each cell contains at most one number (no two numbers in the same cell)
    for r in range(n):
        for c in range(n):
            for v1 in range(n):
                for v2 in range(v1 + 1, n):
                    clauses.append([-varnum(r, c, v1), -varnum(r, c, v2)])

    # Rule 3: Each number appears at most once in each row
    for r in range(n):
        for v in range(n):
            for c1 in range(n):
                for c2 in range(c1 + 1, n):
                    clauses.append([-varnum(r, c1, v), -varnum(r, c2, v)])

    # Rule 4: Each number appears at most once in each column
    for c in range(n):
        for v in range(n):
            for r1 in range(n):
                for r2 in range(r1 + 1, n):
                    clauses.append([-varnum(r1, c, v), -varnum(r2, c, v)])

    return num_variables, clauses


# Example usage:
n = 3  # Order of the Latin square (e.g., 6x6)
num_vars, cnf_clauses = latin_square_to_cnf(n)

# Write the CNF to a file
output_file = 'latin_square.txt'
write_cnf_to_file(num_vars, cnf_clauses, output_file)

print(f"CNF constraints for a {n}x{n} Latin square written to {output_file}")