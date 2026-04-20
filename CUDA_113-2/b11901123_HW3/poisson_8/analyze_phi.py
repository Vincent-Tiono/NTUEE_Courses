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

def main():
    filename = "GPU_8.npy"
    grid_size = 9
    
    phi = read_phi_data(filename, grid_size)
    
    print(f"Successfully read {filename}")
    print(f"Grid shape: {phi.shape}")
    print(f"Min value: {phi.min()}")
    print(f"Max value: {phi.max()}")
    print(f"Value at center: {phi[grid_size//2, grid_size//2, grid_size//2]}")
    
    # Extract potential vs. radius
    distances, potentials = compute_radial_data(phi, grid_size)
    
    # Plot potential vs. distance
    plt.figure(figsize=(10, 6))
    plt.scatter(distances, potentials, alpha=0.5, s=10)
    plt.xlabel('Distance from center (r)')
    plt.ylabel('Potential (φ)')
    plt.title('Potential vs. Distance for L=8')
    
    # In theory, for a point charge we expect φ ∝ 1/r
    # Plot a 1/r curve for comparison
    r_sorted = np.sort(distances)
    ideal = 1/r_sorted
    ideal = ideal * np.max(potentials) / np.max(ideal)  # Scale to match
    plt.plot(r_sorted, ideal, 'r-', label='1/r (scaled)')
    
    plt.legend()
    plt.savefig('poisson_8.png')
    plt.show()

if __name__ == "__main__":
    main()