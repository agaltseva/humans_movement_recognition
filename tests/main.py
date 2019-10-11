import pandas as pd
import numpy as np
import os
from movement_recognition.lapprox.segments import normalize, smooth
import matplotlib.pyplot as plt
from movement_recognition.make_phase_trajectory import phase_track
from movement_recognition.lapprox.models import ssa, ar, arma, base, semor
from movement_recognition.lapprox.joint import joint_mapping
#%%
df = pd.read_csv(os.getcwd()+'/tests/data/Dormitory_f_158_54_36_accm.csv')
df['aT'] = (df['X_value']**2 + df['Y_value']**2 + df['Z_value']**2) ** (1/2)

#%%
# df.head()
# df.shape
#%% 1. GET SEGMENTS
bounds = normalize.extract_segment_bounds(df=df.iloc[:3000], column_name='aT', max_threshold=10, min_segment_size=50)
# print(bounds)
segments = [df.loc[first:second, 'aT'].values for first, second in bounds]
normalized_segments = normalize.normalize_segments(segments)
# print(len(segments))
#%% 3/ GET PHASE TRAJECTORY

track, basis = phase_track(segments[0].reshape(segments[0].shape[0], 1), 20, 2)

# print(track, f'basis: {basis}')
#%%
#
# for b in basis:
#     plt.plot(b)
# plt.show()
# for s in segments:
#     plt.plot(s)
# plt.show()
print(f'track shape:{track[0].shape[0]}',
      f'normalized_segments shape: {normalized_segments[0].shape[0]}')
#%% CREATE LOCAL MODELS
models = {
    "ssa_5": ssa.Ssa(5),
    "ssa_10": ssa.Ssa(10),
    "ar_2": ar.Ar(2),
    "ar_4": ar.Ar(4),
# #     "arma_2_2": arma.Arma(2, 2),
#     "fft_2": fft.Fft(2),
#     "fft_5": fft.Fft(5),
#     "semor_run": semor.Semor(semor_profiles['run']),
#     "semor_walk": semor.Semor(semor_profiles['walk']),
#     "semor_up": semor.Semor(semor_profiles['up']),
#     "semor_down": semor.Semor(semor_profiles['down'])
}
mapper = joint_mapping.JointMapping(models)
X_intermediate1 = mapper.fit_transform(track[0])
X_intermediate2 = mapper.fit_transform(normalized_segments[0])

mapper.namespace_indices


#%%
nparr = df['X_value'].to_numpy()
# print(nparr)
new_inds1 = normalize.shrink_segment(nparr, 10)
new_inds2 = normalize.shrink_segment(nparr)
print(new_inds1)