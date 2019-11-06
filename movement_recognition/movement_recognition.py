from typing import List

import numpy as np
import pandas as pd
from scipy.spatial import distance

from movement_recognition.lapprox.models import ssa, ar
from movement_recognition.lapprox.segments import normalize
from movement_recognition.make_phase_trajectory import to_phase_trajectory


class MovementRecognizer:
    def __init__(self,
                 # models: Dict[str, base.BaseLocalModel],
                 euclidean_col_name: str = 'aT',
                 max_threshold_for_segments_bound_extraction: int = 10,
                 min_segment_size_for_segments_bound_extraction: int = 50, phase_track_length: int = 20,
                 phase_track_n_components: int = 2,
                 range_ssa_lag: tuple = (2, 10),
                 range_ar_degree: tuple = (2, 10),
                 eps: float = 10e-8
                 ):

        self.phase_trajectory = None
        self.euclidean_col_name = euclidean_col_name
        self.max_threshold_for_segments_bound_extraction = max_threshold_for_segments_bound_extraction
        self.min_segment_size_for_segments_bound_extraction = min_segment_size_for_segments_bound_extraction
        self.phase_track_length = phase_track_length,
        self.phase_track_n_components = phase_track_n_components
        # self.models = models
        self.range_ssa_lag = range_ssa_lag
        self.range_ar_degree = range_ar_degree
        self.eps = eps

    def fit(self, ts: pd.DataFrame) -> 'MovementRecognizer':
        bounds = normalize.extract_segment_bounds(df=ts, column_name=self.euclidean_col_name,
                                                  max_threshold=self.max_threshold_for_segments_bound_extraction,
                                                  min_segment_size=self.min_segment_size_for_segments_bound_extraction)

        segments = [ts.loc[first:second, self.euclidean_col_name].values for first, second in bounds]
        # normalized_segments = normalize.normalize_segments(segments)
        segments = [np.array(s) for s in segments]
        trajectories = [to_phase_trajectory(s, 20) for s in segments]
        ssa_coefs = [self.pickup_ssa_model(trajectory) for trajectory in trajectories]
        ar_coefs = [self.pickup_ar_model(trajectory) for trajectory in trajectories]

        models = {}
        for ssa_coef, ar_coef in zip(ssa_coefs, ar_coefs):
            models[f'ssa_{ssa_coef}'] = ssa.Ssa(ssa_coef)
            models[f'ar_{ar_coef}'] = ar.Ar(ar_coef)

        # mapper = joint_mapping.JointMapping(models)

        # X_intermediate1 = mapper.fit_transform(track[0])
        # X_intermediate2 = mapper.fit_transform(normalized_segments[0])

    def pickup_ssa_model(self, trajectory: List[np.array]):  # pickup or select?
        try:
            fitted = [ssa.Ssa(lag=lag).fit(trajectory) for lag in range(self.range_ssa_lag[0], self.range_ssa_lag[1])]
        except:
            print('too much a max value of lag')
        else:
            Ls = [[distance.euclidean(trajectory[j], fitted[i].get_predicted()[j]) for j in range(len(trajectory))]
                  for i in range(len(fitted))]

            medians = [np.median(L) for L in Ls]
            gradient = np.gradient(medians)
            degree_idx = [i for i in range(len(gradient)) if gradient[i] < self.eps][0]

            return range(self.range_ssa_lag[0], self.range_ssa_lag[1])[degree_idx]

    def pickup_ar_model(self, trajectory) -> int:
        try:
            fitted = [ar.Ar(degree=degree).fit(trajectory)
                      for degree in range(self.range_ar_degree[0], self.range_ar_degree[1])]
        except:
            print('too much a max value of degree')
        else:
            Ls = [[distance.euclidean(trajectory[j], fitted[i].get_predicted()[j]) for j in range(len(trajectory))]
                  for i in range(len(fitted))]

            medians = [np.median(L) for L in Ls]
            gradient = np.gradient(medians)
            degree_idx = [i for i in range(len(gradient)) if gradient[i] < self.eps][0]

            return range(self.range_ar_degree[0], self.range_ar_degree[1])[degree_idx]
