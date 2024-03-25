
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from simulation.source import gaussian_source, create_grid
from simulation.instrument import Antenna, Pointing, get_uv_coverage, create_mask
from radionets.simulations.visualize_simulations import plot_source, plot_spectrum, plot_vlba_uv

def ft(img):
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(img)))

def ift(img):
    return np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(sampled_freqs))))

def tf_ft(x, precision = tf.complex128):
    return tf.signal.ifftshift(tf.signal.fft2d(tf.signal.fftshift(tf.cast(x, precision))))

def tf_ift(x, precision = tf.complex128):
    return tf.math.abs(tf.signal.fftshift(tf.signal.ifft2d(tf.signal.fftshift(tf.cast(x, precision)))))

def apply_mask(x, mask):
    return tf.multiply(x,mask)

def DFT_matrix_2d(N):
    i, j = np.fft.fftshift(np.meshgrid(np.arange(N), np.arange(N)))
    A=np.multiply.outer(i.flatten(), i.flatten())
    B=np.multiply.outer(j.flatten(), j.flatten())
    omega = np.exp(-2*np.pi*1J/N)
    W = np.power(omega, A+B)/N
    return W


if __name__ == "__main__":
    
    # create simulated source
    Nx = Ny = 16
    sim_source = gaussian_source(create_grid(Nx , 1)[0])
    plot_source(sim_source, log=False, out_path="./simulated_source.pdf")

    # "perfect" visibilities from np ft
    plot_spectrum(ft(sim_source), out_path="./perfect_visibility.pdf")

    # "perfect" visibilities from tf ft
    visibilities = tf_ft(tf.convert_to_tensor(sim_source))
    plot_spectrum(visibilities, out_path="./perfect_visibility2.pdf")

    # "perfect" reconstruction (should be same as simulated source)
    recons_perfect = tf_ift(visibilities)
    plot_source(recons_perfect, log=False, out_path="./perfect_source.pdf")

    # create inverse DFT to map from visibility->image
    mat_dft = DFT_matrix_2d(Nx)

    # test matrix reconstruction
    flat_visibilities = tf.reshape(visibilities,[Nx*Ny,1])
    matrix_reco = tf.matmul(mat_dft, flat_visibilities)# apply_mask(sampled_visibilities, tf.cast(mask,tf.complex128)))
    plot_source( np.reshape(matrix_reco,(Nx,Ny)), log=False, out_path="./perfect_source2.pdf")

    # create inverse DFT to map from image->visibility
    mat_dft_inv = tf.linalg.inv(mat_dft)

    ###################################################
    # create the inverse DFT, only using real matrices
    # we accomplix this by splitting every oprator into real and imaginary components [Freal, Fimag]
    F =    tf.concat([tf.math.real(mat_dft_inv),tf.math.imag(mat_dft_inv)],axis=0)
    # create matrix to select real components [I,0]
    select_real = tf.concat([tf.eye(Nx*Ny,dtype=tf.double),tf.zeros((Nx*Ny,Nx*Ny),dtype=tf.double)],axis=1)
    # create matrix to select imaginary components [0,I]
    select_img  = tf.concat([tf.zeros((Nx*Ny,Nx*Ny),dtype=tf.double),tf.eye(Nx*Ny,dtype=tf.double)],axis=1)
    # reconstruct "real-ized visibliy matrix"
    # out: [Vreal, Vimag] = [Freal, Fimag]@I
    out = tf.matmul(F,tf.reshape(sim_source,[Nx*Ny,1]))

    # reassemble the true visibility
    out_real =  tf.matmul(select_real,out)
    out_imag =  tf.matmul(select_img, out)
    visiblities2 = tf.reshape(tf.complex(out_real,  out_imag),(Nx,Ny))
    print("2", visiblities2, visibilities)
    # clearly a norm factor is missing but otherwise looks good
    plot_spectrum(visiblities2, out_path="./perfect_visibility3.pdf")



