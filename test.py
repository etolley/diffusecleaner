#pip install dask-ms



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from scipy import misc
# import imageio
# import numpy as np
from simulation.source import gaussian_source, create_grid
from simulation.instrument import Antenna, Pointing, get_uv_coverage, create_mask
#from radionets.simulations.gaussians import gaussian_source, create_grid
from radionets.simulations.visualize_simulations import plot_source, plot_spectrum, plot_vlba_uv
#import dask.array as da
#from daskms import xds_from_table, xds_to_table

def ft(img):
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(img)))

def ift(img):
    return np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(sampled_freqs))))

def tf_ft(x, precision = tf.complex128):
    return tf.signal.ifftshift(tf.signal.fft2d(tf.signal.fftshift(tf.cast(x, precision))))

def tf_ift(x, precision = tf.complex128):
    return tf.math.abs(tf.signal.fftshift(tf.signal.ifft2d(tf.signal.fftshift(tf.cast(x, precision)))))


def apply_mask(x, mask):
    #img = img.copy()
    #x[~mask.astype(bool)] = 0.
    return tf.multiply(x,mask)

if __name__ == "__main__":
    sim_source = gaussian_source(create_grid(64, 1)[0])
    plot_source(sim_source, log=False, out_path="./simulated_source.pdf")
    plot_spectrum(ft(sim_source), out_path="./frequency_distribution.pdf")

    print(sim_source)
    test = tf_ft(tf.convert_to_tensor(sim_source))
    plot_spectrum(test, out_path="./frequency_distribution2.pdf")

    recons = tf_ift(test)
    plot_source(recons, log=False, out_path="./recons_source.pdf")

    ant = Antenna("vlba")
    p = Pointing(-80, 40)
    p.propagate(num_steps=1)
    u, v, steps = get_uv_coverage(p, ant, iterate=False)

    plot_vlba_uv(u, v, out_path="./uv_coverage.pdf")

    mask = create_mask(u, v, size=64)

    sampled_freqs = apply_mask(test, mask)

    recons_sampled = tf_ift(sampled_freqs)
    plot_source(recons_sampled, log=False, out_path="./recons_source_sampled.pdf")

