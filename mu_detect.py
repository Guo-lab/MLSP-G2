import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class Slicer():
    def __init__(self, filtered_mat: np.ndarray, peaks: list):
        self.signal = filtered_mat  # (64, 73214)
        self.signal_dim = self.signal.shape[0] # 64
        self.window_size = 50
        self.signal_peaks = peaks   # (64, #peaks)
    
    def slice(self, idx: int) -> np.ndarray:
        """
        @param idx: the index of the channel. 0-63
        """
        component = self.signal[idx].reshape(-1, 1)
        peaks_indices = self.signal_peaks[idx]
        windows = []
        for index in peaks_indices.flatten():
            # peaks_indices.flatten(): a list of indices
            start_index = max(0, index - self.window_size // 2)
            end_index = min(component.shape[0], index + self.window_size // 2)
            if end_index - start_index < self.window_size:
                # Onlu accept windows with the correct size
                continue
                        
            window = component[start_index:end_index] # (50, 1)
            windows.append(window)
        
        window_matrix_tmp = np.array(windows) # (530, 50, 1)
        window_matrix = np.squeeze(window_matrix_tmp, axis=-1) # (530, 50)
        # print("Window Matrix Shape: ", window_matrix.shape)  # (#peaks, 50)
        return window_matrix
        
        
class MotorUnitDetector:
    def __init__(self, slicer: Slicer):
        self.slicer = slicer
        self.max_clusters = 10
        
        self.window_matrix = []     # (64, #peaks, 50)
        self.feature_matrix = []    # (64, #peaks, 2)
        self.cluster_labels = []    # (64, #peaks, )
        self.num_peaks = []         # (64, )
        self.mu_indices_64ch = []   # (64, 10, #indices)
        self.mean_waveform = []     # (64, 10, 50)
        
    def reduce_dim(self):
        """
        PCA to reduce the dimension of the window matrix
        """
        pca = PCA(n_components=2)
        for each_channel in range(self.slicer.signal_dim):
            win_matrix = self.slicer.slice(each_channel)
            self.num_peaks.append(win_matrix.shape[0])
            self.window_matrix.append(win_matrix) # (64, #peaks, 50)
            self.feature_matrix.append(
                pca.fit_transform(win_matrix)) # (530, 2)
        
        print("Feature Length: ", len(self.feature_matrix))
    
    def evaluate_clusters(self):
        """
        Without a priori knowledge of the number of clusters, 
        so evaluate the clustering performance first
        
        After evaluating the clustering performance,
        we can use the elbow method to find the optimal number of clusters
        which is 10. self.max_clusters
        """
        tmp_max_clusters = 40
        inertia = []
        for n_clusters in range(1, tmp_max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            print("Fitting KMeans with n_clusters: ", self.feature_matrix[0].shape)
            kmeans.fit(self.feature_matrix[0])
            inertia.append(kmeans.inertia_)

        # Plot the elbow curve
        plt.figure(figsize=(10, 6), facecolor='lightgray')  # Adjust figure size and set background color
        plt.plot(range(1, tmp_max_clusters + 1), inertia, marker='o', linestyle='-')  # Change line style
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Inertia')
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')  
        plt.xticks(range(1, tmp_max_clusters + 1, 2))
        plt.tight_layout()
        plt.show()
        
    def cluster(self):
        """
        K-means clustering (better for lower dimension data)
        """
        kmeans = KMeans(n_clusters=self.max_clusters, random_state=42, n_init=10)
        for each_channel in range(self.slicer.signal_dim):
            self.cluster_labels.append(
                kmeans.fit_predict(self.feature_matrix[each_channel]))
          
    def plot_clusters(self):
        plt.figure(figsize=(10, 6), facecolor='lightgray')
        print("Row 0 Dim-Reduced Feature Shape: ", self.feature_matrix[0].shape)
        plt.scatter(self.feature_matrix[0][:, 0], self.feature_matrix[0][:, 1], c=self.cluster_labels[0], cmap='viridis')
        plt.title('K-means Clustering For Component 1')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()

    def detect_mu(self):
        """
        Detect motor units based on the clustering results
        Compute a mean / representative waveform
        Cluster_labels: (64, 530, )
        """
        for each_channel in range(self.slicer.signal_dim):
            tmp_indices = []
            for cluster_idx in range(self.max_clusters): # 0-9
                cluster_indices = np.where(self.cluster_labels[each_channel] == cluster_idx)[0]
                print("Cluster Indices: ", cluster_indices)
                if len(cluster_indices) > 0:
                    tmp_indices.append(cluster_indices)
            
            # 64 channels, each channel has 10 clusters, each cluster has a list of indices
            self.mu_indices_64ch.append(tmp_indices)
            
    def draw_one_mu(self):
        """
        This function will display what one MU looks like
        The MU in the first channel, first cluster
        """
        plt.figure(figsize=(20, 80))
        # 64 * 10 * x
        for each_basis in range(len(self.mu_indices_64ch[0][1])):
            ax = plt.subplot(len(self.mu_indices_64ch[0][1]), 1, each_basis + 1)
            plt.plot(self.slicer.slice(0)[self.mu_indices_64ch[0][1][each_basis]])
            ax.set_title(f"Channel 1 Label 1: Peak NO. {each_basis + 1}")
            
        plt.subplots_adjust(wspace=0.1, hspace=0.9) 
        plt.show()
        
        
    def compute_mean_waveform(self):
        for each_channel in range(self.slicer.signal_dim):
            
            tmp_repre_waveform = []
            for cluster_idx in range(self.max_clusters):
                cluster_indices = np.where(self.cluster_labels[each_channel] == cluster_idx)[0]
                each_group_wave = self.slicer.slice(each_channel)[cluster_indices]
                each_group_wave_array = np.array(each_group_wave)
                # (50, )
                representative_window = np.mean(each_group_wave_array, axis=0)
                tmp_repre_waveform.append(representative_window)
            # (10, 50)
            tmp_repre_waveform_array = np.array(tmp_repre_waveform)
            # 64 * (10, 50)
            self.mean_waveform.append(tmp_repre_waveform_array)
        
        self.mean_waveform = np.array(self.mean_waveform)
        return self.mean_waveform
    
    
    