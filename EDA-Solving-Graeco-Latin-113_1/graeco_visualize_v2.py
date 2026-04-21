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

def parse_graeco_latin_squares(sat_solution, n):
    """
    Parses the SAT solution literals into two Latin squares.
    
    :param sat_solution: List of literals from the SAT solver output.
    :param n: Size of the Latin squares (n x n).
    :return: Two separate Latin squares (numpy arrays).
    """
    # Initialize two empty Latin squares
    latin_square_1 = np.zeros((n, n), dtype=int)
    latin_square_2 = np.zeros((n, n), dtype=int)

    total_cells = n * n * n

    for literal in sat_solution:
        if literal > 0:
            # Convert to 0-indexed
            literal -= 1
            
            # Determine which square the literal belongs to
            if literal < total_cells:
                square = 1
            else:
                square = 2
                literal -= total_cells  # Adjust literal for second square
                
            # Calculate row, column, and value
            cell = literal // n
            row = cell // n
            col = cell % n
            value = literal % n
            
            # Assign to the correct square (adding 1 to convert back to 1-indexing)
            if square == 1:
                latin_square_1[row, col] = value + 1
            else:
                latin_square_2[row, col] = value + 1

    return latin_square_1, latin_square_2

def visualize_graeco_latin_square_from_file(file_path, n=4):
    """
    Visualizes a Graeco-Latin square based on the SAT solver's output file.
    
    :param file_path: Path to the SAT solver output file.
    :param n: Size of the Graeco-Latin square (default is 4x4).
    :return: Prints and returns the two Latin squares and the combined Graeco-Latin square.
    """
    # Step 1: Parse the SAT solution from the file
    sat_solution = parse_sat_output(file_path)
    
    # Step 2: Parse the solution into two Latin squares
    latin_square_1, latin_square_2 = parse_graeco_latin_squares(sat_solution, n)
    
    # Step 3: Create the Graeco-Latin square by combining the two Latin squares
    graeco_latin_square = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            graeco_latin_square[i, j] = f"({latin_square_1[i, j]}, {latin_square_2[i, j]})"
    
    # Step 4: Display the squares
    print("First Latin Square:")
    print(latin_square_1)
    
    print("\nSecond Latin Square:")
    print(latin_square_2)
    
    print("\nGraeco-Latin Square (Combined):")
    print(graeco_latin_square)
    
    return latin_square_1, latin_square_2, graeco_latin_square

# Example usage:
output_file_path = '/Users/vincent_tiono/Documents/EDA/graeco_latin_output.txt'  # Path to your SAT solver output file
visualize_graeco_latin_square_from_file(output_file_path, n=4)