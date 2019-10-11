
'''
1. Find bounds
2. Creating segments
2. Align segments
'''
from typing import Iterable, List
import numpy as np
import pandas as pd
from movement_recognition.align_time import normalize_segments



def find_local_extrema(array: np.ndarray, radius: int):
    output = np.array([False] * len(array))
    for idx, val in enumerate(array):
        indexes = np.array([x
                            for x in range(idx - radius, idx + radius + 1)
                            if x > 0 and x < len(array)])
        output[idx] = array[idx] >= np.max(array[indexes])
    return output

def extract_segment_bounds(df,
                           column_name,
                           max_threshold,
                           min_segment_size,
                           max_segment_size=10e8,
                           radius=1):
    # We take only those which have left and right neighbour
    indices = list(df.index)[1:-1]
    idx_local_maxima = np.where(find_local_extrema(df['aT'].values, radius))[0]
    idx_above_threshold = (idx for idx in idx_local_maxima if df.loc[idx, column_name] > max_threshold)
    idx_maxima = list(idx_above_threshold)
    idx_segments = [(idx_maxima[i], idx_maxima[i + 1]) for i in range(len(idx_maxima) - 1)]
    big_enough_segments = [pair for pair in idx_segments
                           if (pair[1] - pair[0] >= min_segment_size
                               and pair[1] - pair[0] <= max_segment_size)]
    return big_enough_segments

def create_segments(df: pd.DataFrame, col_name: str = 'aT', max_threshold: int = 10,
                    min_segment_size: int=50, max_segment_size=10e8, radius=1) -> List[np.ndarray]:
    bounds = extract_segment_bounds(df=df, column_name=col_name, max_threshold=max_threshold,
                                    min_segment_size=min_segment_size, max_segment_size=max_segment_size,
                                    radius=radius)
    # print(bounds)
    return [df.loc[first:second, 'aT'].values for first, second in bounds]









# %% PENKIN`s CODE

#
# def get_data(name):
#     data = pd.read_csv(rel_path + name).iloc[500:-500, :]
#     data["R"] = np.sqrt(data["X"]**2 + data["Y"]**2 + data["Z"]**2)
#     data = data["R"]
#     data.index = np.arange(data.shape[0])
#     return data
#
#
# def segement_generator(data: pd.DataFrame, segment_length: int):
#     for o in range(segment_length, data.shape[0],
#                    segment_length):
#         segment = data.iloc[o - segment_length: o]
#         segment.index = np.arange(segment.shape[0])
#         yield segment


# %% GRABOVOI`s CODE
# def return_h(input, i, l=10):
#     return np.sum(input[:, i:i + l, :], axis=-1)
#
#
# def return_phase_track(input, l=10):
#     """
#     input has a shape [batch_size, time_len, 1]
#     """
#
#     phase_track = np.zeros([input.shape[0], input.shape[1] - l, l])
#
#     for i in range(0, input.shape[1] - l):
#         phase_track[:, i, :] = return_h(input, i, l)
#
#     return phase_track[0]
#
# def find_points(points, line_point):
#     """
#     points have a shape [N x 2]
#     line_point has a shape [2 x 1]
#     """
#     List_of_points_plus = []
#     List_of_points_minus = []
#
#     List_of_t_plus = []
#     List_of_t_minus = []
#
#     for i in range(len(points) - 1):
#         if (line_point[1] * points[i][0] - line_point[0] * points[i][1] < 0) and (
#                 line_point[1] * points[i + 1][0] - line_point[0] * points[i + 1][1] > 0):
#             List_of_points_plus.append(points[i])
#             List_of_t_plus.append(i)
#         if (line_point[1] * points[i][0] - line_point[0] * points[i][1] > 0) and (
#                 line_point[1] * points[i + 1][0] - line_point[0] * points[i + 1][1] < 0):
#             List_of_points_minus.append(points[i])
#             List_of_t_minus.append(i)
#
#     return np.array(List_of_points_plus), np.array(List_of_points_minus), np.array(List_of_t_plus), np.array(
#         List_of_t_minus)
#
#
# def find_distance(points, line_point):
#     """
#     points have a shape [N x 2]
#     line_point has a shape [2 x 1]
#     """
#
#     sum_distance = 0
#
#     normal = np.array([line_point[1], -line_point[0]])
#     normal = normal / np.sqrt((normal * normal).sum())
#
#     for p in points:
#         sum_distance += ((normal * p).sum())
#
#     return sum_distance
#
# def find_segment(X, T):
#     phase_track = return_phase_track(X, T)
#     model = PCA(n_components=2)
#
#     ress = model.fit_transform(phase_track)
#
#     ress[:, 0] = ress[:, 0] / np.sqrt(((ress[:, 0] ** 2).mean()))
#
#     ress[:, 1] = ress[:, 1] / np.sqrt(((ress[:, 1] ** 2).mean()))
#
#     Phi = np.linspace(-np.pi, np.pi, 200)
#
#     All_List = np.array(list(map(lambda phi: find_points(ress, np.array([np.sin(phi), np.cos(phi)])), Phi)))
#
#     List_of_std = []
#     for l, phi in zip(All_List, Phi):
#         List_of_std.append(find_distance(np.vstack([l[0], l[1]]), np.array([np.sin(phi), np.cos(phi)])))
#
#     List_of_std = np.array(List_of_std)
#
#     phi = Phi[np.argmin(List_of_std)]
#
#     line_point = np.array([np.sin(phi), np.cos(phi)])
#
#     List_of_points_plus, List_of_points_minus, List_of_t_plus, List_of_t_minus = find_points(ress, line_point)
#
#     return List_of_points_plus, List_of_points_minus, List_of_t_plus, List_of_t_minus, line_point, ress
#
#
# def segmentation(X_all, prediction_vector, T):
#     List_of_point = []
#     List_of_All = []
#     for t in np.unique(prediction_vector):
#         ind = np.where(prediction_vector == t)[0]
#
#         X = X_all[:, ind, :]
#         List_of_t = np.arange(0, X.shape[1], 1)
#
#         List_of_points_plus, List_of_points_minus, List_of_t_plus, List_of_t_minus, line_point, ress = find_segment(X,T)
#
#         List_of_All.append(
#             [X, List_of_t, List_of_points_plus, List_of_points_minus, List_of_t_plus, List_of_t_minus, line_point,
#              ress])
#         List_of_point.append((np.where(prediction_vector == t)[0])[List_of_t_minus])
