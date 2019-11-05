import pickle
# import sklearn.metrics
import pandas as pd
import numpy as np
import os
from scipy.spatial import distance
from sklearn import metrics

from movement_recognition.lapprox.segments import normalize, smooth
import matplotlib.pyplot as plt
from movement_recognition.make_phase_trajectory import to_phase_trajectory, phase_track
import  movement_recognition
from movement_recognition.lapprox.models import ssa, ar, fft, base
from movement_recognition.lapprox.joint import joint_mapping
#%%
#Part 0: read and preproc data
df = pd.read_csv(os.getcwd()+'/data/Dormitory_f_158_54_36_accm.csv') #/tests
df['aT'] = (df['X_value']**2 + df['Y_value']**2 + df['Z_value']**2) ** (1/2)
df_sit = df[500: 3100].reset_index(drop=True)
df_mug = df[5000:8100].reset_index(drop=True)
#%% PLOT BLOCK
#%%
# a = df['X_value'].plot()
# a2 = df['Y_value'].plot()
# a3 = df['Z_value'].plot()
# a4 = df['aT'].plot()
# plt.legend([a, a2, a3, a4], ['X', 'Y', 'Z', 'aT'])
# df_mug.plot()
# plt.show()
# column_name='X_value'
# print(df_mug.loc[6050:6100, column_name])
#%% 1. GET SEGMENTS
# Segments becomes the same length

column_name='X_value'

bounds = normalize.extract_segment_bounds(df=df_mug, column_name=column_name,
                                          min_threshold=-2, min_segment_size=800)
print('bounds:', bounds)
segments = [df_mug.loc[first:second, column_name].values for first, second in bounds]
normalized_segments = normalize.normalize_segments(segments)
#%%
segments = [np.array(s) for s in segments]

#%% 3/ GET PHASE TRAJECTORY
range_models_params = (10, 15)
trajectories = [to_phase_trajectory(s, 20) for s in segments]

print('tr len', len(trajectories[0]))
# fitted = []
# for param in range(range_models_params[0], range_models_params[1]):
#         ar_ = ar.Ar(param)
#         fitted.append(ar_.fit(trajectories[0]))

# t = [[(i, j) for i in range(3) ] for j in range(100, 103)]
fitted = [ar.Ar(degree=lag).fit(trajectories[0]) for lag in range(range_models_params[0], range_models_params[1])]
print('lol')
Ls = [[distance.euclidean(trajectories[0][j], fitted[i].get_predicted()[j]) for j in range(len(trajectories[0]))]
         for i in range(len(fitted)) ]
print('Ls[0]', Ls[0], f'len Ls: {len(Ls)}')

#%%
a = [2, 5]
print(np.linspace(a)[1])

# grad = np.gradient([np.median(L) for L in Ls])
# print(len(Ls), len(grad))


# print('trajectory[0]', trajectories[0][0])
# print('predicted',res[0].get_predicted()[0])
# [print(f'i={i}', f'j={j}', distance.euclidean(trajectories[0][j], res[i].get_predicted()[j]))  for j in range(1004) for i in range(len(res))]
      # 'f1:', sklearn.metrics.f1_score(trajectories[0][0], res[0].get_predicted()[0]))


# plt.plot(list(trajectories[0][0]))
# plt.plot(list(res[7].get_predicted()[0]))
# plt.show()
#%% CREATE LOCAL MODELS
models = {
    "ssa_5": ssa.Ssa(5),
    "ssa_10": ssa.Ssa(10),
    "ar_2": ar.Ar(2),
    "ar_4": ar.Ar(4),
#     "arma_2_2": arma.Arma(2, 2),
    "fft_2": fft.Fft(2),
    "fft_5": fft.Fft(5),
#     "semor_run": semor.Semor(semor_profiles['run']),
#     "semor_walk": semor.Semor(semor_profiles['walk']),
#     "semor_up": semor.Semor(semor_profiles['up']),
#     "semor_down": semor.Semor(semor_profiles['down'])
}
#%%
# mapper = joint_mapping.JointMapping(models)
# interp = mapper.fit(track)
# # transformed = mapper.transform()
# print('track[0]:', track, 'interp', interp)#, 'transformed:', transformed)
# # X_intermediate1 = mapper.fit_transform(track[0])
# # X_intermediate2 = mapper.fit_transform(normalized_segments[0])
# #
#
# #%%
#
# interp = mapper.fit_transform(track)

#%%
# with open("data/semor_profiles.pickle", "rb") as f:
#     semor_profiles = pickle.load(f)
#
# for key, profile in semor_profiles.items():
#     semor_profiles[key] = normalize.shrink_segment(profile, 20)
#
# with open("data/segments.pickle", "rb") as f:
#     segments_dict = pickle.load(f)
#
#
# action_and_segment = []
#
# for key, segments_array in segments_dict.items():
#     action_and_segment.extend([(key, s) for s in segments_array if s.size > 30])
#
# segments = [pair[1] for pair in action_and_segment]
# actions = [pair[0] for pair in action_and_segment]
#
# normalized_segments = normalize.normalize_segments(segments)
# action_and_norm_segment = [(pair[0][0], pair[1])
#                            for pair in zip(action_and_segment, normalized_segments)]
# le = sklearn.preprocessing.LabelEncoder()
# Y = le.fit_transform(actions)
# X = np.array(normalized_segments)



# print(normalized_segments[0])
# track, basis = phase_track(segments, 20)#segments[0].reshape(segments[0].shape[0], 1), 20, 2)
# track = track.reshape(1, 6)
# print('track', track.shape)
# print('basis', basis.shape)
# print(track, f'basis: {basis}')



#
# for b in track:
# plt.plot(track[0])
# plt.title('track')
# plt.show()
# for s in segments:
#     plt.plot(s)
# print(f'track shape:{track[0].shape[0]}')

# #%%
# print(track)
# c = 0
# a = ar.Ar(degree=3)
# a.fit(track)
# a.transform(track)