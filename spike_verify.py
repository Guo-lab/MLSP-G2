import numpy as np
from collections import deque
import sys
import copy
from mlsp_utils.utils_converter import *
import matplotlib.pyplot as plt


class SpikeVerifyer:
    def __init__(
        self, verified_label_matrix: list, mu_indices_64ch: list, 
        peak_indices: list, filted_mat: np.array):
        
        self.label = copy.deepcopy(verified_label_matrix) # 63 x 10 x (63-self[0])
        self.mu_indices_64ch = mu_indices_64ch
        self.peak_indices = peak_indices
        self.signals = filted_mat
        self.num_ch = len(mu_indices_64ch)
        
        self.multi_MU_labels = [] # flatten all channels, label 0 in channel is label 10
        self.mu_spikes = []
        self.interval = []
        
    def merge_labels_based_MU(self):
        # a DFS / BFS is needed. 
        
        for each_ch_compare in range(len(self.label)): # 0-62
            # each row is the current channel, and what is stored in 2D is which label in which channel is similar
            curr_ch = self.label[each_ch_compare] # (10, 63 - self[0])]
            for each_lbl_in_curr_ch in range(len(curr_ch)): # 0-9
                
                tmp_lbl_pair = []
                if self.label[each_ch_compare][each_lbl_in_curr_ch] is None:
                    continue
                match_lbl = curr_ch[each_lbl_in_curr_ch]
                match_lbl_list = list(map(int, match_lbl.flatten()))
                match_ch_list = list(
                    range(each_ch_compare + 1, len(match_lbl_list) + each_ch_compare + 1))
                # from 1 to 63
                
                tmp_lbl_pair.append((each_ch_compare, each_lbl_in_curr_ch))
                tmp_lbl_pair.extend(self.collect_dfs(match_ch_list, match_lbl_list))
                
                # when > 1, there are 68 MUs. 
                # Hyperparameter to decide how many identical MUs are needed
                if len(tmp_lbl_pair) > 2:
                    self.multi_MU_labels.append(tmp_lbl_pair)
            
    def collect_dfs(self, ch: list, lbl: list) -> list:
        assert len(ch) == len(lbl)
        if len(lbl) == 1:
            return []
        
        ret = []
        for each in range(len(lbl)):
            if lbl[each] == -1:
                continue
            if ch[each] >= self.num_ch - 1:
                continue
            if self.label[ch[each]][lbl[each]] is None:
                continue
            
            match_lbl = self.label[ch[each]][lbl[each]]
            match_lbl_list = list(map(int, match_lbl.flatten()))
            match_ch_list = list(
                range(ch[each] + 1, len(match_lbl_list) + ch[each] + 1))
            
            ret.append((ch[each], lbl[each]))
            ret.extend(self.collect_dfs(match_ch_list, match_lbl_list))
            self.label[ch[each]][lbl[each]] = None
        
        return ret
        
            
    def get_MU_spikes_interval(self):
        print("There are", len(self.multi_MU_labels), "identical detected MUs in total.")
        for each_MU in self.multi_MU_labels:
            spikes_idx = []
            cnt = 0
            for ch, lbl in each_MU:   
                mu_peak_indices = self.mu_indices_64ch[ch][lbl]
                spikes_idx.extend(self.peak_indices[ch][mu_peak_indices])
                # For each channel, get the indices of this MU's spikes
                
            print("There are", len(spikes_idx), "spikes in total.")
            spikes_idx.sort()
            self.mu_spikes.append(spikes_idx)
            
            tmp_span = peaks2timespan(spikes_idx, self.signals.shape[1])
            self.interval.append(tmp_span)
            
            
    def plot_MU_spikes_interval(self):
        cmap = plt.get_cmap('rainbow')
        plt.figure(figsize=(10, 40))

        for each_basis in range(len(self.interval)):
            ax = plt.subplot(len(self.interval), 1, each_basis + 1)
            color = cmap(each_basis / len(self.interval))
            
            signal_background = np.mean(self.signals, axis=0)            
            plt.plot(signal_background, color="lightgrey")
            plt.axis('off')
            for i, each_interval in enumerate(self.interval[each_basis]):
                plt.axvspan(
                    each_interval[0], 
                    each_interval[1], 
                    color=color, 
                    alpha=1)
                
            ax.set_title(f"Spikes: Motor Unit No. {each_basis + 1}")
        
        
        plt.subplots_adjust(hspace=1) 
        plt.show()
            
    
    
    
""" draft
Approximately, if find a similar label in other channels, merge them,
and do not need to reverse that label again.
Otherwise, a DFS / BFS is needed.

def merge_labels_based_MU(self):
    for each_ch_compare in range(len(self.label)): # 0-62
        curr_ch = self.label[each_ch_compare] # (10, 63 - self[0])]
        for each_lbl_in_curr_ch in range(len(curr_ch)): # 0-9
            
            tmp_lbl_pair = []
            if self.label[each_ch_compare][each_lbl_in_curr_ch] is None:
                continue
            match_lbl = curr_ch[each_lbl_in_curr_ch]
            match_lbl_list = list(map(int, match_lbl.flatten()))
            match_ch_list = list(
                range(each_ch_compare + 1, len(match_lbl_list) + each_ch_compare + 1))
            
            tmp_lbl_pair.append((each_ch_compare, each_lbl_in_curr_ch))
            
            for each in range(len(match_lbl_list)):
                if match_lbl_list[each] == -1:
                    continue
                tmp_lbl_pair.append((match_ch_list[each], match_lbl_list[each]))
                self.label[match_ch_list[each] - 1][match_lbl_list[each]] = None
            
            self.multi_MU_labels.append(tmp_lbl_pair)
"""
