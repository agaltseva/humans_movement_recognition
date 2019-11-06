from typing import Iterable, Optional, List, Tuple

import numpy as np
import scipy.interpolate
from scipy import interpolate


def shrink_segment(segment: np.ndarray, output_length: int) -> np.ndarray:
    segment_indices = np.arange(segment.size)
    interpolated_f = interpolate.interp1d(segment_indices,
                                          segment,
                                          kind='cubic')
    new_indices = np.linspace(0, segment_indices[-1], output_length)
    return interpolated_f(new_indices)


def normalize_segments(segments: Iterable[np.ndarray],
                       length: Optional[int] = None) -> List[np.ndarray]:
    segments = list(segments)
    min_segment_size = min(segment.size for segment in segments)
    length = length if (length and min_segment_size>length) else min_segment_size
    return [shrink_segment(segment, length) for segment in segments]


def dumb_smooth(
        segments: Iterable[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Trims segments to the minimum length and returns the mean and std.

    :param segments: Iterable[np.ndarray]. Iterable of segments.
    :return: Pair of np.ndarrays.
    """
    segments = list(segments)
    min_length_segment = min(segments, key=len)
    min_length = len(min_length_segment)
    segments = [segment[:min_length] for segment in segments]
    segments_matrix = np.vstack(segments)
    return (np.mean(segments_matrix, axis=0),
            np.std(segments_matrix, axis=0))


def cubic_smooth(segments: Iterable[np.ndarray],
                 output_length: Optional[int]=None) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates mean and std of passed segments. Since the sizes might vary,
    the cubic spline interpolation is performed.

    :param segments: Iterable[np.ndarray]. Iterable of segments.
    :param output_length: The desired length of the segment to convert to.
    :return: Pair of np.ndarrays.
    """
    segments = list(segments)

    if not output_length:
        lengths = [len(s) for s in segments]
        output_length = sorted(lengths)[len(lengths) // 2]

    segments_matrix = np.zeros((len(segments), output_length))
    output_indices = np.arange(output_length)
    normalized_segments = normalize_segments(segments, length=output_length)
    for idx, segment in enumerate(normalized_segments):
        # segment_indices = np.arange(len(segment))
        # f = scipy.interpolate.interp1d(segment_indices, segment, kind='cubic')
        #       TODO: WHAT DOES THIS CODE DO?
        # extrapolate_indices = (output_indices *
        #                        (segment_indices[-1] / output_indices[-1]))
        # extrapolate_indices[-1] = segment_indices[-1]

        segments_matrix[idx, :] = segment #f(extrapolate_indices)

    return (np.mean(segments_matrix, axis=0),
            np.std(segments_matrix, axis=0))