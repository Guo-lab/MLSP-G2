from dtw import *
import fastdtw
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .utils_constants import *
from tqdm import tqdm


def calculate_dtw_distance(series1, series2):
    distance = fastdtw.fastdtw(series1, series2)
    return distance[0]


# Normal DTW
#   Reference: @https://dynamictimewarping.github.io/python/#the-alignment-class

#NOTE: notes from the former 18797 TA's 
#   not cross-correlation
#   not correlation

# Final evaluation metric statemnet
#    Intersection hyperparameter
#   distance of the peaks hyperparameter
#   threshold of the similarities hyperparameter
#   range of valid firing instance moments

#NOTE, 2024: 
# Now, the similarity matrix is attained by calculating the signals distance,
# Maybe the similarity could be measured by the PCA results


def compute_similarity_matrix(multi_signals: np.array, mode: str) -> tuple:
    matrix_shape = (
        multi_signals.shape[0], multi_signals.shape[1],
        multi_signals.shape[0], multi_signals.shape[1]
    )
    dtw_similarity = np.zeros(matrix_shape)
    cos_similarity = np.zeros(matrix_shape)
    cov_similarity = np.zeros(matrix_shape)
    
    for channel in tqdm(range(multi_signals.shape[0]), desc="Channels"):
        for label in range(multi_signals.shape[1]):
            
            # The baseline is multi_signals[ch, label]
            for other_ch in range(multi_signals.shape[0]):
                if other_ch <= channel:
                    continue
                
                for other_ch_label in range(multi_signals.shape[1]):
                    tmp_dtw_similarity = calculate_dtw_distance(
                        multi_signals[channel, label], 
                        multi_signals[other_ch, other_ch_label]
                    )
                    dtw_similarity[channel, label, other_ch, other_ch_label] = tmp_dtw_similarity
                    
                    if label == 0 and other_ch_label == 0 and channel == 0 and other_ch == 1:
                        # Plot one for DTW distance
                        alignment = dtw(
                            multi_signals[channel, label], 
                            multi_signals[other_ch, other_ch_label], 
                            keep_internals=True)
                        dtw(multi_signals[channel, label],
                            multi_signals[other_ch, other_ch_label],
                            keep_internals=True, 
                            step_pattern=rabinerJuangStepPattern(6, "c")
                        ).plot(type="twoway",offset=-2)
                        
                    # Cosine Similarity
                    tmp_cos_similarity = cosine_similarity(
                        multi_signals[channel, label].reshape(1, -1), 
                        multi_signals[other_ch, other_ch_label].reshape(1, -1)
                    )
                    cos_similarity[channel, label, other_ch, other_ch_label] = tmp_cos_similarity[0][0]
                    
                    # Covariance Similarity
                    tmp_cov_similarity = np.corrcoef(
                        multi_signals[channel, label], 
                        multi_signals[other_ch, other_ch_label]
                    )
                    cov_similarity[channel, label, other_ch, other_ch_label] = tmp_cov_similarity[0][1]
    
                    # print("====================================")
                    # print("DTW Similarity: ", tmp_dtw_similarity)
                    # print("Cosine Similarity: ", tmp_cos_similarity)
                    # print("Covariance Similarity: ", tmp_cov_similarity)
                    # print("====================================")
                    print(f"[{channel}:{label}]<>[{other_ch}:{other_ch_label}]", 
                          tmp_dtw_similarity, tmp_cos_similarity[0][0], tmp_cov_similarity[0][1])
    
               
    # save as bin 
    if mode == 'steady':
        np.save(SIMILARITY_PATH + 'dtw_matrix_4D.npy', dtw_similarity)
        np.save(SIMILARITY_PATH + 'cos_matrix_4D.npy', cos_similarity)
        np.save(SIMILARITY_PATH + 'cov_matrix_4D.npy', cov_similarity)
    elif mode == 'incr_1':
        np.save(SIMILARITY_PATH + 'dtw_matrix_4D_incr_1.npy', dtw_similarity)
        np.save(SIMILARITY_PATH + 'cos_matrix_4D_incr_1.npy', cos_similarity)
        np.save(SIMILARITY_PATH + 'cov_matrix_4D_incr_1.npy', cov_similarity)
    elif mode == 'incr_2':
        np.save(SIMILARITY_PATH + 'dtw_matrix_4D_incr_2.npy', dtw_similarity)
        np.save(SIMILARITY_PATH + 'cos_matrix_4D_incr_2.npy', cos_similarity)
        np.save(SIMILARITY_PATH + 'cov_matrix_4D_incr_2.npy', cov_similarity)
        
    return dtw_similarity, cos_similarity, cov_similarity



def merge_similar_label(similarity_matrix: np.array, threshold: float) -> list:
    """
    Merge similar labels based on the similarity matrix
    """
    # Initialize the cluster labels
    list_of_similar_labels = []
    for each_ch in range(similarity_matrix.shape[0]):
        if each_ch == similarity_matrix.shape[0] - 1:
            break
        list_of_similar_for_each_label = []
        for label in range(similarity_matrix.shape[1]):
            # For each lable in baseline channel, 
            # find the most similar label in other channels 
            # (#num of channels (64 - ch - 1))
            matching_MU_label_from_each_channel = np.zeros((similarity_matrix.shape[0] - each_ch - 1, 1))
            for other_ch in range(similarity_matrix.shape[2]):
                if other_ch <= each_ch:
                    continue
                # Threshold of the similarities would be hyperparameter
                # Merge two labels in other channels 
                # if cos is higher than a threshold
                highest_cov = -1
                for label_other_ch in range(similarity_matrix.shape[3]):
                    if similarity_matrix[each_ch, label, other_ch, label_other_ch] > highest_cov:
                        highest_cov = similarity_matrix[each_ch, label, other_ch, label_other_ch]
                        if highest_cov <= threshold:
                            matching_MU_label_from_each_channel[other_ch - each_ch - 1] = -1
                            continue
                        if other_ch == 1 and each_ch == 0 and label == 0:
                            pass
                        
                        highest_cov_label = label_other_ch
                        matching_MU_label_from_each_channel[other_ch - each_ch - 1] = highest_cov_label
            
            
            list_of_similar_for_each_label.append(matching_MU_label_from_each_channel)
     
        list_of_similar_labels.append(list_of_similar_for_each_label)
    return list_of_similar_labels

    