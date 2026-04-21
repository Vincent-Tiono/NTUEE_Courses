import numpy as np

def parse_sat_output(file_path):
    """
    Parse the SAT solver output from the file and return the list of literals.
    
    :param file_path: Path to the SAT output file.
    :return: List of literals from the file.
    """
    literals = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('SAT'):
                continue
            literals += map(int, line.strip().split())
    return literals

def visualize_latin_square_from_file(file_path, n=100):
    """
    Visualizes a Latin square based on the SAT solver's output file.
    
    :param file_path: Path to the SAT solver output file.
    :param n: Size of the Latin square (default is 10x10).
    :return: Prints and returns the Latin square.
    """
    # Parse the SAT solution from the file
    sat_solution = parse_sat_output(file_path)

    # Initialize the Latin square grid
    latin_square = np.zeros((n, n), dtype=int)
    
    # Parse the solution to find the positive literals
    for literal in sat_solution:
        if literal > 0:  # Only consider positive literals (which represent true variables)
            # Map literal to (row, col, value)
            literal -= 1  # Convert to 0-indexed
            r = literal // (n * n)
            c = (literal % (n * n)) // n
            v = (literal % n) + 1
            latin_square[r, c] = v
    
    print("Latin Square:")
    print(latin_square)
    return latin_square

# Example usage:
output_file_path = 'latin_output.txt'  # Path to your SAT solver output file
visualize_latin_square_from_file(output_file_path, n=50)
