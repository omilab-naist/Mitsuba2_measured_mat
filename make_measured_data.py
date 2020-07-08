import csv
import pprint

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

import time
from numba import jit
from tqdm import tqdm

import os
import math
from save_tensor import saveTensor

file_name = 'Al#4_Gaussian blur_15'
save_file_name = 'm_mat_test'
os.makedirs('output\\' + save_file_name, exist_ok=True)
def main():
    #ones((各次元の長さ))
    theta_i     = np.ones((2),dtype='float32')
    phi_i       = np.ones((2),dtype='float32')
    ndf         = np.ones((2, 2),dtype='float32')
    sigma       = np.ones((2, 2),dtype='float32')
    vndf        = np.ones((2, 2, 2, 2),dtype='float32')
    spectra     = np.ones((2, 2, 2, 2, 2),dtype='float32')
    luminance   = np.ones((2, 2, 2, 2),dtype='float32')
    wavelengths = np.ones((2),dtype='float32')
    description = np.ones((2),dtype='uint8')
    jacobian    = np.ones((1),dtype='uint8') #shapeは必ず1

    m_data = {'theta_i':theta_i, 'phi_i':phi_i, 'ndf':ndf, 'sigma':sigma, 'vndf':vndf, 'spectra':spectra, 'luminance':luminance, 'wavelengths':wavelengths, 'description':description, 'jacobian':jacobian}
    saveTensor('Output\\' + save_file_name + '\\m_mat_test',m_data)    

if __name__ == "__main__":
    main()
