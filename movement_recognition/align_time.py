from typing import List, Tuple

import numpy as np
from scipy import interpolate

def dumb_align(time: np.ndarray) -> np.ndarray:
    ticks = time.size
    period = (time[-1] - time[0])/(ticks - 1)
    new_time = np.arange(time[0], time[-1]+period, period)
    return new_time

def cubic_align(time: np.ndarray,
                values_list: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
    time_aligned = dumb_align(time)
    
    #for interpolation
    mask = time[1:] != time[:-1]
    mask = np.hstack([np.array([True]), mask])
    
    new_values_list = []
    for values in values_list:
        f = interpolate.interp1d(time[mask],
                                 values[mask],
                                 kind='cubic',
                                 fill_value="extrapolate")
        new_values = f(time_aligned)
        new_values_list.append(new_values)
    return (time_aligned, new_values_list)