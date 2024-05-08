def peaks2timespan(peak_moments, filted_steady_mat_width):
    """
    Given peak moments indices (vector), and 
    filtered steady mat width (time series length e.g. 700000) as the maximum boundry
    
    Return a list of tuples, 
    each tuple is a time span pair (start_index, end_index)
    
    Time span is window size of 50, 
    centered at the peak moment, as the form of a pair above.
    
    (Ignore the overlap between two time spans, 
    which just increases the opacity of the time span in the plot 
    if any (overlap).)
    """
    window_size = 50
    list_indices = []
    
    for i in range(len(peak_moments)):
        start_index = max(0, peak_moments[i] - window_size // 2)
        end_index = min(
            filted_steady_mat_width, peak_moments[i] + window_size // 2)
        
        start_index = int(start_index)
        end_index = int(end_index)
        list_indices.append((start_index, end_index))
        
    list_indices = list(set(list_indices))
    return list_indices