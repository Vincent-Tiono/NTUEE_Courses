import numpy as np

def fftreal(x, y):
    """
    Compute the FFT of two N-point real signals x and y using only one N-point FFT.
    
    Parameters:
    x, y : array_like
        Input real signals of length N
    
    Returns:
    Fx, Fy : ndarray
        FFT of x and FFT of y respectively
    """
    # Ensure inputs are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Check that both signals have the same length
    if len(x) != len(y):
        raise ValueError("Both signals must have the same length")
    
    N = len(x)
    
    # Create complex signal z = x + j*y
    z = x + 1j * y
    
    # Compute FFT of the complex signal
    Z = np.fft.fft(z)
    
    # Initialize output arrays
    Fx = np.zeros(N, dtype=complex)
    Fy = np.zeros(N, dtype=complex)
    
    # Handle DC component (k=0)
    Fx[0] = Z[0].real
    Fy[0] = Z[0].imag
    
    # Handle Nyquist frequency (k=N/2) for even N
    if N % 2 == 0:
        Fx[N//2] = Z[N//2].real
        Fy[N//2] = Z[N//2].imag
    
    # Handle other frequency components
    for k in range(1, (N+1)//2):
        
        Fx[k] = 0.5 * (Z[k] + np.conj(Z[N-k]))
        Fy[k] = -0.5j * (Z[k] - np.conj(Z[N-k]))
        
        # Use conjugate symmetry to fill the negative frequencies
        if k != N-k:  # Avoid double assignment when k = N-k
            Fx[N-k] = np.conj(Fx[k])
            Fy[N-k] = np.conj(Fy[k])
    
    return Fx, Fy

# Example usage and verification
if __name__ == "__main__":
    # Create test signals
    N = 8
    n = np.arange(N)
    
    # Test signals
    x = np.cos(2 * np.pi * n / N) + 0.5 * np.cos(4 * np.pi * n / N)
    y = np.sin(2 * np.pi * n / N) + 0.3 * np.sin(6 * np.pi * n / N)
    
    # Compute FFT using our function
    Fx, Fy = fftreal(x, y)
    
    # Verify by computing FFTs separately
    Fx_ref = np.fft.fft(x)
    Fy_ref = np.fft.fft(y)
    
    print("\nOriginal signals:")
    print("x =", x)
    print("y =", y)
    
    print("\nFFT results:")
    print("Fx =", Fx)
    print("Fy =", Fy)
    
    print("\nSeparate FFTs:")
    print("Fx_ref =", Fx_ref)
    print("Fy_ref =", Fy_ref)
    print("\n")