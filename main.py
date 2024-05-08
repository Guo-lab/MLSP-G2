# In[1]:
import sys
sys.dont_write_bytecode = True

import numpy as np
import matplotlib.pyplot as plt
import h5py
from mlsp_utils.utils_filter import *
from mlsp_utils.utils_constants import *
from mlsp_utils.utils_peak import AMPD
from mlsp_utils.utils_similarity import compute_similarity_matrix
from mlsp_utils.utils_similarity import merge_similar_label
from mlsp_utils.utils_converter import peaks2timespan

from preprocess import Preprocessor
from peak_detect import PeakDetector
from mu_detect import Slicer, MotorUnitDetector
from spike_verify import SpikeVerifyer


## In[2]: 
##### Preprocess the data #####
preprocessor = Preprocessor()
filted_steady_mat, filted_increase_mat_one, filted_increase_mat_two = preprocessor.preprocess_mat()
# preprocessor.preprocess_display()

## In[3]:
##########################
##### Peak Detection #####
##########################
"""
         /\
        /  \
       /    \       /\ 
      /      \     /  \
     /        \   /    \
    /          \ /      \
   /            X        \
  /            / \        \
 /            /   \        \
/            /     \        \
----------------------------------> Time (or Position)
"""
## AMPD slow to run: 2 * 64 mins
"""
all_peak_indices = []
for each_base in range(filted_steady_mat.shape[0]):
    peak_indices = AMPD(filted_steady_mat[each_base])
    all_peak_indices.append(np.array(peak_indices).reshape(-1, 1))
"""
## Use built-in function to find peaks
peak_detector = PeakDetector(filted_steady_mat, filted_increase_mat_one, filted_increase_mat_two)
peak_detector.detect_peaks()
# peak_detector.display_peaks()
"""peak_detector.peak_indices_steady: (64, #peaks)"""

## In[4]:
### Slice the windows for each peak moment ###
slicer = Slicer(filted_steady_mat, peak_detector.peak_indices_steady)
steady_mu_detector = MotorUnitDetector(slicer)


## In[5]:
###### PCA to reduce the dimension 
###### K-Means to cluster 
###### get the numbers of the Motor Units / peak windows
steady_mu_detector.reduce_dim()
# steady_mu_detector.evaluate_clusters()

# In[6]:
# Choose K = 10
steady_mu_detector.cluster()
steady_mu_detector.plot_clusters()


## In[7]:
steady_mu_detector.detect_mu()
steady_mu_detector.draw_one_mu()

## In[8]:
waveform_64channel = steady_mu_detector.compute_mean_waveform()


# In[23]:
# compute_similarity_matrix(waveform_64channel, mode="steady")


# ##### Pick up the matching MUs

# In[27]:
loaded_dtw = np.load(SIMILARITY_PATH + 'dtw_matrix_4D.npy')
loaded_cos = np.load(SIMILARITY_PATH + 'cos_matrix_4D.npy')
loaded_cov = np.load(SIMILARITY_PATH + 'cov_matrix_4D.npy')

# In[28]:
verified_label_matrix = merge_similar_label(loaded_cos, 0.99) # 63 x 10 x (63-self[0])





###########################################################################
############# For the first MU in the first & second channel ##############
###########################################################################
# In[30]:
###### Finding the firing moments for MU one in channel one
##### Put each matching MU into plot and find out the firing moments
steady_spikes = SpikeVerifyer(
    verified_label_matrix, 
    steady_mu_detector.mu_indices_64ch,
    peak_detector.peak_indices_steady, 
    filted_steady_mat)
# steady_mu_detector.mu_indices_64ch: 
#   peak indices corresponding to the label in one channel
# peak_detector.peak_indices_steady:
#   peak's real indices in all channels
steady_spikes.merge_labels_based_MU()

# In[31]:
steady_spikes.get_MU_spikes_interval()

# In[35]:
steady_spikes.plot_MU_spikes_interval()

###### Compare the first MU(peak waveform) in the first channel 
# with this kind of MUs in all other 63 channels.
# MU_indices_in_other_ch_from_ch_1_lbl_1 = get_MU_indices_in_other_ch(
#     steady_mu_detector.mu_indices_64ch, verified_label_matrix[0][0])


# final_span_mu_1 = []
# final_span_ch_1 = peaks2timespan(
#     np.squeeze(
#         peak_detector.peak_indices_steady[0][steady_mu_detector.mu_indices_64ch[0][0]]
#     ).tolist(), 
#     filted_steady_mat.shape[1])
# final_span_mu_1.append(final_span_ch_1)

# for other_channels in range(len(MU_indices_in_other_ch_from_ch_1_lbl_1)):
#     ifValid = MU_indices_in_other_ch_from_ch_1_lbl_1[other_channels]
#     print(ifValid)
#     if not isinstance(ifValid, np.ndarray) and ifValid == -1:
#         print("No matching MU1 found in this channel", other_channels + 1)
#         final_span_mu_1.append(-1)
#         continue
    
#     peaks_this_channel = np.squeeze(peak_detector.peak_indices_steady[other_channels + 1][MU_indices_in_other_ch_from_ch_1_lbl_1[other_channels]])
#     final_span_this_channel = peaks2timespan(
#         peaks_this_channel, 
#         filted_steady_mat.shape[1]
#     )
#     final_span_mu_1.append(final_span_this_channel)


# In[36]:
# Plot
# plt.figure(figsize=(10, 100))

# for each_basis in range(filted_steady_mat.shape[0]):
#     ax = plt.subplot(filted_steady_mat.shape[0], 1, each_basis + 1)
#     plt.plot(filted_steady_mat[each_basis])
    
#     if final_span_mu_1[each_basis] == -1:
#         # No matching MU (peak waveform) found in this channel
#         ax.set_title(f"Channel(Basis) NO. {each_basis + 1}")
#         continue
    
#     for start, end in final_span_mu_1[each_basis]:
#         plt.axvspan(start, end, color='red', alpha=0.3)
        
#     ax.set_title(f"Channel(Basis) NO. {each_basis + 1}")
    
# plt.subplots_adjust(wspace=0.1, hspace=0.9) 
# plt.show()



#NOTE: 
# From a strict perspective, the intersection of the red areas should be considered, 
# In a relaxed view, the constraint be less, the union is considered.
# The standard is just different.






# In[36]:
#### For MUs in the first channel, 
# find their matching MUs in other channels respectively.





# In[32]:
# ch_2_peaks = peak_detector.peak_indices_steady[1]
# ch2_peaks_for_ch1_lbl1 = np.squeeze(ch_2_peaks[MU_indices_in_other_ch_from_ch_1_lbl_1[0]])
# final_span_ch2 = peaks2timespan(
#     ch2_peaks_for_ch1_lbl1, filted_steady_mat.shape[1])

# ch_1_peaks = peak_detector.peak_indices_steady[0]
# final_span_ch1 = peaks2timespan(
#     np.squeeze(
#         ch_1_peaks[steady_mu_detector.mu_indices_ch[0][0]]
#     ).tolist(), 
#     filted_steady_mat.shape[1])

# # In[34]:
##### Plot the result above
# plt.figure(figsize=(20, 5))
# for each_basis in range(2):
#     ax = plt.subplot(2, 1, each_basis + 1)
#     plt.plot(filted_steady_mat[each_basis])
#     if each_basis == 0:
#         cnt = 0
#         for start, end in final_span_ch1:
#             cnt = cnt + 1
#             plt.axvspan(start, end, color='red', alpha=0.3)
#         print(cnt)
#     elif each_basis == 1:   
#         cntt = 0
#         for start, end in final_span_ch2:
#             cntt = cntt + 1
#             plt.axvspan(start, end, color='red', alpha=0.3)
#         print(cntt)
#     ax.set_title(f"Channel(Basis) NO. {each_basis + 1}")
# plt.subplots_adjust(wspace=0.1, hspace=0.4) 
# plt.show()
###########################################################################
###########################################################################




# In[37]:
############## Increase force 1 ##############
slicer_increase_one = Slicer(filted_increase_mat_one, peak_detector.peak_indices_incr_1)
incr_one_detector = MotorUnitDetector(slicer_increase_one)
incr_one_detector.reduce_dim()
incr_one_detector.cluster()
incr_one_detector.detect_mu()
waveform_64channel_incr_one = incr_one_detector.compute_mean_waveform()
compute_similarity_matrix(waveform_64channel_incr_one, mode="incr_1")

loaded_dtw = np.load(SIMILARITY_PATH + 'dtw_matrix_4D_incr_1.npy')
loaded_cos = np.load(SIMILARITY_PATH + 'cos_matrix_4D_incr_1.npy')
loaded_cov = np.load(SIMILARITY_PATH + 'cov_matrix_4D_incr_1.npy')
verified_label_matrix = merge_similar_label(loaded_cos, 0.99)

incr_1_spikes = SpikeVerifyer(
    verified_label_matrix, 
    incr_one_detector.mu_indices_64ch,
    peak_detector.peak_indices_incr_1, 
    filted_increase_mat_one)
incr_1_spikes.merge_labels_based_MU()
incr_1_spikes.get_MU_spikes_interval()
incr_1_spikes.plot_MU_spikes_interval()

# In[38]:
############## Increase force 2 ##############
slicer_increase_two = Slicer(filted_increase_mat_two, peak_detector.peak_indices_incr_2)
incr_two_detector = MotorUnitDetector(slicer_increase_two)
incr_two_detector.reduce_dim()
incr_two_detector.cluster()
incr_two_detector.detect_mu()
waveform_64channel_incr_two = incr_two_detector.compute_mean_waveform()
compute_similarity_matrix(waveform_64channel_incr_two, mode="incr_2")

loaded_dtw = np.load(SIMILARITY_PATH + 'dtw_matrix_4D_incr_2.npy')
loaded_cos = np.load(SIMILARITY_PATH + 'cos_matrix_4D_incr_2.npy')
loaded_cov = np.load(SIMILARITY_PATH + 'cov_matrix_4D_incr_2.npy')
verified_label_matrix = merge_similar_label(loaded_cos, 0.99)

incr_2_spikes = SpikeVerifyer(
    verified_label_matrix, 
    incr_two_detector.mu_indices_64ch,
    peak_detector.peak_indices_incr_2, 
    filted_increase_mat_two)
incr_2_spikes.merge_labels_based_MU()
incr_2_spikes.get_MU_spikes_interval()
incr_2_spikes.plot_MU_spikes_interval()

