def generate_graeco_latin_cnf(n, pre_assigned=None):
    """
    Generate CNF clauses for a Graeco-Latin square of order n with pre-assigned values.
    
    Args:
    - n: size of the Graeco-Latin square
    - pre_assigned: list of tuples (row, col, val1, val2) for pre-assigned values (1-indexed)
    
    Returns:
    - The number of variables and clauses
    - The CNF clauses as a list of lists
    """
    clauses = []
    num_vars = 2 * n * n * n  # n^3 variables for two Latin squares

    # Helper function to convert row, col, val for the Latin square to a variable
    def var_latin(row, col, val, latin_idx):
        return (row * n * n) + (col * n) + val + 1 + (latin_idx * n * n * n)

    # Check for invalid pre-assigned values
    if pre_assigned:
        for row, col, val1, val2 in pre_assigned:
            # Ensure row and col are within bounds, and val1, val2 are between 1 and n
            if not (1 <= row <= n and 1 <= col <= n and 1 <= val1 <= n and 1 <= val2 <= n):
                print(f"Error: Invalid pre-assigned value at (row={row}, col={col}, val1={val1}, val2={val2})")
                return None, None  # Return None to signal an error

    # Latin square constraints for both squares
    for latin_idx in range(2):
        for row in range(n):
            for col in range(n):
                # Ensure each cell contains exactly one number
                clauses.append([var_latin(row, col, val, latin_idx) for val in range(n)])

                # Each number can only appear once per row and column in the square
                for val1 in range(n):
                    for val2 in range(val1 + 1, n):
                        # No two different values in the same cell
                        clauses.append([-var_latin(row, col, val1, latin_idx), -var_latin(row, col, val2, latin_idx)])
                    
                    # No duplicate values in the same row
                    for col2 in range(n):
                        if col != col2:
                            clauses.append([-var_latin(row, col, val1, latin_idx), -var_latin(row, col2, val1, latin_idx)])
                    
                    # No duplicate values in the same column
                    for row2 in range(n):
                        if row != row2:
                            clauses.append([-var_latin(row, col, val1, latin_idx), -var_latin(row2, col, val1, latin_idx)])

    # Orthogonality constraint: Ensure each pair of (Latin1_val, Latin2_val) appears exactly once in the entire square
    for val1 in range(n):
        for val2 in range(n):
            # Ensure that this pair (val1 in Latin 1, val2 in Latin 2) appears at least once
            clauses.append([var_latin(row, col, val1, 0) and var_latin(row, col, val2, 1) for row in range(n) for col in range(n)])
            
            # Ensure that this pair appears at most once
            for row1 in range(n):
                for col1 in range(n):
                    for row2 in range(n):
                        for col2 in range(n):
                            if row1 != row2 or col1 != col2:
                                clauses.append([
                                    -var_latin(row1, col1, val1, 0),
                                    -var_latin(row1, col1, val2, 1),
                                    -var_latin(row2, col2, val1, 0),
                                    -var_latin(row2, col2, val2, 1)
                                ])

    # Add pre-assigned values
    if pre_assigned:
        for row, col, val1, val2 in pre_assigned:
            # Adjust for 0-based indexing
            row, col = row - 1, col - 1
            val1, val2 = val1 - 1, val2 - 1
            clauses.append([var_latin(row, col, val1, 0)])
            clauses.append([var_latin(row, col, val2, 1)])

    return num_vars, clauses

def write_cnf_file(num_vars, clauses, filename="graeco_latin_square.txt"):
    if num_vars is None or clauses is None:
        print("CNF file generation skipped due to invalid pre-assigned values.")
        return
    
    with open(filename, 'w') as f:
        # Write the problem line
        f.write(f"p cnf {num_vars} {len(clauses)}\n")
        # Write the clauses
        for clause in clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")

# Example usage: Generate the CNF for a 4x4 Graeco-Latin square with pre-assigned values
n = 4
pre_assigned = [(1, 1, 1, 2), (2, 2, 1, 3)]  # The second pre-assigned value is invalid (5 > 4)
num_vars, clauses = generate_graeco_latin_cnf(n, pre_assigned)
write_cnf_file(num_vars, clauses)