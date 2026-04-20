import numpy as np
import matplotlib.pyplot as plt
import argparse

def freq_samp(k: int):
    """
    Implements frequency sampling method for FIR filter design.
    
    Args:
        k (int): Determines filter length N = 2k + 1
    """
    # Calculate filter length
    N = 2 * k + 1

    # Initialize desired frequency response
    Hd = np.zeros(N, dtype=complex)
    
    # Create desired frequency response
    for i in range(N):
        if i/N < 0.5:
            Hd[i] = 2j * np.pi * i/N
        else:
            Hd[i] = 2j * np.pi * (i-N)/N

    # Handle transition band (points k and k+1)
    slope = (Hd[k+2] - Hd[k-1])/3
    Hd[k] = Hd[k-1] + slope 
    Hd[k+1] = Hd[k+2] - slope

    # Calculate impulse response using inverse FFT
    h = np.fft.ifft(Hd)

    # Reorder impulse response for causal filter
    r = np.zeros(N, dtype=complex)
    r[0:k] = h[k+1:N]
    r[k:] = h[0:k+1]

    # Plot impulse response
    plt.figure(figsize=(10,8))
    plt.subplot(211)
    plt.stem(r, markerfmt='C0o', linefmt='C0-', basefmt='C0-') 
    plt.title("FIR Filter Impulse Response (N = {})".format(N))
    plt.xlabel("Time Index (n)")
    plt.ylabel("Amplitude")
    plt.xticks(np.arange(0, N))
    plt.grid(True, alpha=0.3)

    # Calculate frequency response
    F = np.linspace(0, 1, 10001)
    R = np.zeros(len(F), dtype=complex)
    H = np.zeros(len(F), dtype=complex)

    # Compute frequency response for each frequency point
    for i in range(len(F)):
        # Ideal frequency response
        if F[i] < 0.5: 
            H[i] = 2j * np.pi * F[i]
        else:
            H[i] = 2j * np.pi * (F[i]-1)
        
        # Compute actual frequency response
        for j in range(N):
            R[i] += r[j] * np.exp(-1j * 2 * np.pi * F[i] * (j-k))

    # Plot frequency response
    plt.subplot(212) 
    plt.plot(F, R.imag, 'C3-', label="Designed Filter", linewidth=2)
    plt.plot(F, H.imag, 'C5-', label="Ideal Filter", linewidth=2)
    plt.title("Frequency Response Comparison")
    plt.legend(loc='upper right')
    plt.xlabel("Normalized Frequency (F)")
    plt.ylabel("Imaginary Part")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to parse command line arguments and run frequency sampling.
    """
    parser = argparse.ArgumentParser(description='FIR Filter Design using Frequency Sampling Method')
    parser.add_argument('--k', type=int, default=19, 
                       help='Determines filter length N = 2k + 1 (default: 19)')
    
    args = parser.parse_args()
    freq_samp(args.k)    

if __name__ == "__main__":
    main()