import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def Wf(F_val):
    # Weight function:
    if F_val >= 0.3:
        w = 1
    elif F_val <= 0.25:
        w = 0.6
    else:
        w = 0  # In the transition region, use stopband weight
    return w

def hw01():
    fs = 8000                   # Sampling frequency 8000 Hz
    N = 19                      # Filter Length
    k = (N - 1) // 2            # Midpoint index = 9 for N=19
    delta = 0.0001              # Delta in Step 5

    # STEP 01: Initialization
    # Frequency sample points (in normalized frequency, where 0.5 corresponds to 4000 Hz)
    F = np.array([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5])
    # Desired sample response: highpass filter: passband if F >= 0.3, stopband otherwise.
    H = (F >= 0.3).astype(float)
    M = len(F)  # M = 11

    # Frequency axis for evaluation (normalized frequency from 0 to 0.5)
    f = np.arange(0, 0.5 + delta, delta)
    # Desired frequency response: Hd = 0 for f < 0.275; 1 for f >= 0.275.
    Hd = (f >= 0.275).astype(float)

    # Weighting function:
    # W = 0.6 for stopband (f <= 0.25) and 1 for passband (f >= 0.3)
    W1 = 1.0 * (f >= 0.3)
    W2 = 0.6 * (f <= 0.25)
    W = W1 + W2

    n = 0
    E1 = 99
    E0 = 5

    # Main iteration loop; continue until convergence (|E1 – E0| <= delta)
    while abs(E1 - E0) > delta:
        # Update M and H in case F has been modified by peak detection
        M = len(F)
        H = (F >= 0.3).astype(float)
        
        # STEP 02: Build matrix A (size M×M)
        # Use M-1 cosine terms and one extra column for the weighted term.
        A = np.empty((M, M))
        for i in range(M):
            A[i, :M-1] = np.cos(2 * np.pi * np.arange(M-1) * F[i])
            A[i, M-1]  = ((-1) ** i) / Wf(F[i])
        
        # Solve linear system for S (S is an M-element vector)
        S = np.linalg.solve(A, H)
        
        # STEP 03: Compute frequency response RF using cosine expansion
        RF = np.zeros_like(f)
        for i in range(M-1):
            RF += S[i] * np.cos(2 * i * np.pi * f)
        
        err = (RF - Hd) * W
        
        # STEP 04: Local peak detection (code kept for clarity even if redundant)
        F_temp = F.copy()
        q = 1  # index starts from 1 since F_temp[0] is fixed at 0
        for i in range(1, len(f) - 1):
            if err[i] > err[i-1] and err[i] > err[i+1]:
                if q < len(F_temp):
                    F_temp[q] = delta * i
                    q += 1
            elif err[i] < err[i-1] and err[i] < err[i+1]:
                if q < len(F_temp):
                    F_temp[q] = delta * i
                    q += 1
        
        # STEP 05: Update error measure and frequency vector F using peak detection
        E1 = E0
        n += 1
        max_locs, _ = find_peaks(err)
        min_locs, _ = find_peaks(-err)
        E0 = np.max(np.abs(err))
        # Output the maximal error for the current iteration
        print(f"Iteration {n}: Max error = {E0:.6f}")
        F_new = np.sort(np.concatenate(([0],
                                         (max_locs * delta).astype(float),
                                         (min_locs * delta).astype(float),
                                         [0.5])))
        F = F_new

    # STEP 06: Create symmetric impulse response h from S
    h = np.zeros(N)
    h[k] = S[0]
    for i in range(1, k+1):
        h[k+i] = S[i] / 2
        h[k-i] = S[i] / 2
        
    # Plot Frequency Response
    f_Hz = f  # Convert normalized frequency to Hz (0 to 4000 Hz)
    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.plot(f_Hz, RF, 'k', label='RF')
    plt.plot(f_Hz, Hd, 'b', label='Hd')
    plt.title('Frequency Response')
    plt.xlabel('Frequency (Hz)')
    plt.legend()

    # Plot Impulse Response
    plt.subplot(212)
    x = np.arange(0, N)
    plt.stem(x, h, basefmt=" ")
    plt.axhline(0, color='r', linestyle='-')
    plt.title('Impulse Response')
    plt.xlabel('Time index')
    plt.xlim([-1, N])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    hw01()