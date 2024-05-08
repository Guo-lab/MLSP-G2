import numpy as np
from tqdm import tqdm 


def AMPD(data):  # Slow to run
    """
    param data: 1-D numpy.ndarray 
    return:     peak indices
    
    Reference:  https://zhuanlan.zhihu.com/p/549588865
    """
    p_data = np.zeros_like(data, dtype=np.int32)
    count = data.shape[0]
    arr_rowsum = []
    
    for k in tqdm(range(1, count // 2 + 1)):
        row_sum = 0
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                row_sum -= 1
                
        arr_rowsum.append(row_sum)
        
    min_index = np.argmin(arr_rowsum)
    max_window_length = min_index
    
    for k in tqdm(range(1, max_window_length + 1)):
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                p_data[i] += 1
                
    return np.where(p_data == max_window_length)[0]

