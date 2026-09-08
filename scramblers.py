import numpy as np
from scipy.interpolate import UnivariateSpline
def scramble_signal(v, smoothing=1000, block_size=20, seed=None):
    rng = np.random.default_rng(seed)

    # Time axis: samples 0, 1, 2, ...
    t = np.arange(len(v))

    # Fit smooth spline
    spline = UnivariateSpline(t, v, s=smoothing)
    smooth = spline(t)

    # Difference between actual signal and smooth signal
    residual = v - smooth

    # Permute residuals in blocks rather than individual samples
    n_blocks = len(residual) // block_size
    blocks = residual[:n_blocks * block_size].reshape(n_blocks, block_size)

    # Shuffle the blocks
    rng.shuffle(blocks)

    # Put them back together
    shuffled_residual = blocks.ravel()

    # Deal with any remainder
    remainder = residual[n_blocks * block_size:]
    shuffled_residual = np.concatenate([
        shuffled_residual,
        remainder
    ])

    # Reconstruct
    scrambled = smooth + shuffled_residual

    return scrambled

import numpy as np


def phase_scramble(v, max_shift_samples=10, seed=None):
    """
    Perturb Fourier phases using a random temporal shift for each frequency.

    Parameters
    ----------
    v : np.ndarray
        1D time-series signal.
    max_shift_samples : float
        Maximum equivalent temporal shift, in samples.
        Each frequency receives an independent random shift
        between -max_shift_samples and +max_shift_samples.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Phase-perturbed signal.
    """

    rng = np.random.default_rng(seed)

    # Fourier transform
    V = np.fft.rfft(v)

    amplitude = np.abs(V)
    phase = np.angle(V)

    # Frequencies expressed as cycles/sample
    freqs = np.fft.rfftfreq(len(v))

    # Independent temporal shift for every frequency
    shifts = rng.uniform(
        -max_shift_samples,
        max_shift_samples,
        len(freqs)
    )

    # Convert sample shift -> phase shift
    phase_shift = 2 * np.pi * freqs * shifts

    # Don't alter the DC component (mean)
    phase_shift[0] = 0

    # Apply perturbation
    new_phase = phase + phase_shift

    # Reconstruct Fourier representation
    V_new = amplitude * np.exp(1j * new_phase)

    # Back to time domain
    return np.fft.irfft(V_new, n=len(v))
