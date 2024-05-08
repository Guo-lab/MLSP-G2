"""
    The file to preprocess the force data
    High-Density Electromyography (HDEMG)
    
    Import the .mat signal and use the helper function
    provided by the MLSP TA to filter the data 
    
    
    Two matrix per file:
    -	Out_mat: a 64 x # of samples array, 
        with the time series data 
        corresponding to the HDEMG signals
    -	Grid_crds: a 64 x 2 array, 
        with the spatial location of each electrode, 
        the unit is in mm. 
        This is useful while planning to 
        incorporate spatial filters in our method. 
        
"""
import sys
sys.dont_write_bytecode = True

import numpy as np
import h5py
import matplotlib.pyplot as plt
from mlsp_utils.utils_filter import *
from mlsp_utils.utils_constants import *


class Preprocessor():
    def __init__(self):
        self.filted_increase_mat_one = None
        self.filted_increase_mat_two = None
        self.filted_steady_mat = None

    def preprocess_mat(self):
        """
        Preprocess the .mat file. Grid Matrix not returned here.
        @return filted_steady_mat, steady force samples array, filtered signals with the time series.
        @return filted_increase_mat_one, increasing force 1 samples array.
        @return filted_increase_mat_two, increasing force 2 samples array.
        """
        with h5py.File(INCR_FORCE_1, 'r') as file:
            increase_grid_one = file['grid_crds'][()].T
            increase_mat_one = file['out_mat'][()].T
            
        with h5py.File(INCR_FORCE_2, 'r') as file:
            increase_grid_two = file['grid_crds'][()].T
            increase_mat_two = file['out_mat'][()].T
            
        with h5py.File(STEADY_FORCE, 'r') as file:
            steady_grid = file['grid_crds'][()].T
            steady_mat = file['out_mat'][()].T
            
        filted_increase_mat_one = filt_GRID(increase_mat_one)
        filted_increase_mat_two = filt_GRID(increase_mat_two)
        filted_steady_mat = filt_GRID(steady_mat)
        
        assert filted_increase_mat_one.shape == increase_mat_one.shape 
        print("Increase force 1 matrix shape: ", filted_increase_mat_one.shape)
        assert filted_increase_mat_two.shape == increase_mat_two.shape
        print("Increase force 2 matrix shape: ", filted_increase_mat_two.shape)
        assert filted_steady_mat.shape == steady_mat.shape
        print("Steady force matrix shape: ", filted_steady_mat.shape)
        
        print("increase_one Grid_crds shape: ", increase_grid_one.shape)
        print("increase_two Grid_crds shape: ", increase_grid_two.shape)
        print("steady Grid_crds shape: ", steady_grid.shape)
        
        print("Preprocessing done!")
        
        np.savetxt(
            PREPROCESSED_PATH + 'increase_mat_one.csv',
            filted_increase_mat_one, 
            delimiter=','
        )

        np.savetxt(
            PREPROCESSED_PATH + 'increase_mat_two.csv',
            filted_increase_mat_two,
            delimiter=','
        )

        np.savetxt(
            PREPROCESSED_PATH + 'steady_mat.csv',
            filted_steady_mat,
            delimiter=','
        )

        self.filted_increase_mat_one = filted_increase_mat_one
        self.filted_increase_mat_two = filted_increase_mat_two
        self.filted_steady_mat = filted_steady_mat
        
        return filted_steady_mat, filted_increase_mat_one, filted_increase_mat_two
    

    def preprocess_display(self):
        """
        Display the preprocessed data, 3 columns for each basis.
        """
        print("Display the preprocessed signals.")
        num_basis = self.filted_steady_mat.shape[0]      
        
        plt.figure(figsize=(30, 70))

        for each_basis in range(num_basis):
            ax = plt.subplot(num_basis, 3, 3 * each_basis + 1)
            plt.plot(self.filted_steady_mat[each_basis], color='red')
            ax.set_title(f"Comp NO. {each_basis + 1}")

            ax = plt.subplot(num_basis, 3, 3 * each_basis + 2)
            plt.plot(self.filted_increase_mat_one[each_basis], color='blue')
            ax.set_title(f"Comp NO. {each_basis + 1}")

            ax = plt.subplot(num_basis, 3, 3 * each_basis + 3)
            plt.plot(self.filted_increase_mat_two[each_basis], color='orange')
            ax.set_title(f"Comp NO. {each_basis + 1}")
            
        plt.subplots_adjust(wspace=0.1, hspace=0.9)
        plt.show()

        