from scipy.signal import find_peaks
import matplotlib.pyplot as plt


class PeakDetector():
    def __init__(self, signals_matrix_1, signals_matrix_2, signals_matrix_3):
        self.name = 'peak detector'
        self.peak_indices_steady = []
        self.peak_indices_incr_1 = []
        self.peak_indices_incr_2 = []
        self.signals_steady = signals_matrix_1
        self.signals_incr_1 = signals_matrix_2
        self.signals_incr_2 = signals_matrix_3
        self.num_components = signals_matrix_1.shape[0]
        
    def detect_peaks(self):
        for each_basis in range(self.num_components):
            tmp_peaks, _ = find_peaks(
                self.signals_steady[each_basis], height=0, distance=100, threshold=None)
            tmp_peaks = tmp_peaks.reshape(-1, 1)
            self.peak_indices_steady.append(tmp_peaks)
            
            if each_basis == 0:
                print("Steady: For the first basis, peaks indices.shape: ", tmp_peaks.shape)
            if each_basis == 1:
                print("Steady: For the second basis, peaks indices.shape: ", tmp_peaks.shape)
                
            tmp_peaks, _ = find_peaks(
                self.signals_incr_1[each_basis], height=0, distance=100, threshold=None)
            tmp_peaks = tmp_peaks.reshape(-1, 1)
            self.peak_indices_incr_1.append(tmp_peaks)
            
            tmp_peaks, _ = find_peaks(
                self.signals_incr_2[each_basis], height=0, distance=100, threshold=None)
            tmp_peaks = tmp_peaks.reshape(-1, 1)
            self.peak_indices_incr_2.append(tmp_peaks)
    
        assert len(self.peak_indices_steady) == self.num_components
        
        
    def display_peaks(self):
        plt.figure(figsize=(30, 70))
        
        for each_basis in range(self.num_components):
            tmp_peaks_steady = self.peak_indices_steady[each_basis]
            tmp_peaks_incr_1 = self.peak_indices_incr_1[each_basis]
            tmp_peaks_incr_2 = self.peak_indices_incr_2[each_basis]
            
            ax = plt.subplot(self.num_components, 3, 3 * each_basis + 1)
            plt.plot(self.signals_steady[each_basis], color='g')
            plt.scatter(tmp_peaks_steady, self.signals_steady[each_basis][tmp_peaks_steady], marker='x', color='r')
            ax.set_title(f"Comp NO. {each_basis + 1}")
            
            ax = plt.subplot(self.num_components, 3, 3 * each_basis + 2)
            plt.plot(self.signals_incr_1[each_basis], color='b')
            plt.scatter(tmp_peaks_incr_1, self.signals_incr_1[each_basis][tmp_peaks_incr_1], marker='x', color='r')
            ax.set_title(f"Comp NO. {each_basis + 1}")
            
            ax = plt.subplot(self.num_components, 3, 3 * each_basis + 3)
            plt.plot(self.signals_incr_2[each_basis], color='orange')
            plt.scatter(tmp_peaks_incr_2, self.signals_incr_2[each_basis][tmp_peaks_incr_2], marker='x', color='r')
            ax.set_title(f"Comp NO. {each_basis + 1}")
        
        plt.subplots_adjust(wspace=0.1, hspace=1) 
        plt.show()