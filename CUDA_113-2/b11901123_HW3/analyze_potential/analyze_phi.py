import numpy as np
import matplotlib.pyplot as plt

def read_phi_data(filename, grid_size):
    """Read the binary data file and reshape to 3D grid"""
    # Read binary file with float32 values (matches the CUDA code's float type)
    data = np.fromfile(filename, dtype=np.float32)
    
    # Reshape to 3D grid
    phi = data.reshape((grid_size, grid_size, grid_size))
    return phi

def compute_radial_data(phi, grid_size):
    """Compute potential vs. distance from center"""
    # Calculate center of the grid
    center = grid_size // 2
    
    # Create arrays to store distance and potential
    distances = []
    potentials = []
    
    # Calculate distance and potential for each point
    for z in range(grid_size):
        for y in range(grid_size):
            for x in range(grid_size):
                # Calculate distance from center
                r = np.sqrt((x-center)**2 + (y-center)**2 + (z-center)**2)
                
                # Only include points that aren't at the center (source point)
                if r > 0.1:
                    distances.append(r)
                    potentials.append(phi[z, y, x])
    
    return np.array(distances), np.array(potentials)

def analyze_single_grid(filename, grid_size):
    """Analyze a single grid file and return the data"""
    try:
        phi = read_phi_data(filename, grid_size)
        
        print(f"Successfully read {filename}")
        print(f"Grid shape: {phi.shape}")
        print(f"Min value: {phi.min()}")
        print(f"Max value: {phi.max()}")
        print(f"Value at center: {phi[grid_size//2, grid_size//2, grid_size//2]}")
        
        # Extract potential vs. radius
        distances, potentials = compute_radial_data(phi, grid_size)
        return distances, potentials
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None, None

def main():
    # Define different grid sizes and filenames
    grid_configs = [
        {"filename": "GPU_8.npy", "grid_size": 9, "color": "blue", "label": "L=8"},
        {"filename": "GPU_16.npy", "grid_size": 17, "color": "red", "label": "L=16"},
        {"filename": "GPU_32.npy", "grid_size": 33, "color": "green", "label": "L=32"},
        {"filename": "GPU_64.npy", "grid_size": 65, "color": "purple", "label": "L=64"}
    ]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot data for each grid size
    for config in grid_configs:
        distances, potentials = analyze_single_grid(config["filename"], config["grid_size"])
        
        if distances is not None and potentials is not None:
            plt.scatter(distances, potentials, alpha=0.5, s=10, 
                      color=config["color"], label=config["label"])
    
    # In theory, for a point charge we expect φ ∝ 1/r
    # Plot a 1/r curve for comparison (using the largest distance range)
    max_distances = []
    max_potentials = []
    for config in grid_configs:
        distances, potentials = analyze_single_grid(config["filename"], config["grid_size"])
        if distances is not None and distances.size > 0:
            max_distances.extend(distances)
            max_potentials.extend(potentials)
    
    if max_distances:
        r_range = np.linspace(0.5, max(max_distances), 1000)
        ideal = 1/r_range
        # Scale to match approximately the average maximum potential
        scale_factor = np.mean([np.max(potentials) for _, potentials in 
                              [(analyze_single_grid(c["filename"], c["grid_size"])) 
                               for c in grid_configs if analyze_single_grid(c["filename"], c["grid_size"])[0] is not None]])
        ideal = ideal * scale_factor / np.max(ideal)
        plt.plot(r_range, ideal, 'k-', label='1/r (scaled)')
    
    plt.xlabel('Distance from center (r)', fontsize=14)
    plt.ylabel('Potential (φ)', fontsize=14)
    plt.title('Potential vs. Distance for Different Grid Sizes', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('poisson_comparison.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()