from typing import Iterable, List, Optional
from movement_recognition.lapprox.misc import local_extrema
import numpy as np
from scipy import interpolate

def extract_segment_bounds(df,
                           column_name,
                           max_threshold,
                           min_segment_size,
                           max_segment_size=10e8,
                           radius=1):
    # We take only those which have left and right neighbour
    indices = list(df.index)[1:-1]
    idx_local_maxima = np.where(local_extrema.find_local_extrema(df['aT'].values, radius))[0]
    idx_above_threshold = (idx for idx in idx_local_maxima if df.loc[idx, column_name] > max_threshold)
    idx_maxima = list(idx_above_threshold)
    idx_segments = [(idx_maxima[i], idx_maxima[i + 1]) for i in range(len(idx_maxima) - 1)]
    big_enough_segments = [pair for pair in idx_segments
                           if (pair[1] - pair[0] >= min_segment_size
                               and pair[1] - pair[0] <= max_segment_size)]
    return big_enough_segments

def shrink_segment(segment: np.ndarray, output_length: int) -> np.ndarray:
    segment_indices = np.arange(segment.size)
    interpolated_f = interpolate.interp1d(segment_indices,
                                                segment,
                                                kind='cubic')
    new_indices = np.linspace(0, segment_indices[-1], output_length)
    return interpolated_f(new_indices)


def normalize_segments(segments: Iterable[np.ndarray],
                       length: Optional[int]=None) -> List[np.ndarray]:
    segments = list(segments)
    length = length if length else min(segment.size for segment in segments)
    return [shrink_segment(segment, length) for segment in segments]
