# import pickle
# import sklearn.metrics
import os

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from movement_recognition.lapprox.joint import joint_mapping
from movement_recognition.lapprox.models import ssa, ar, fft
from movement_recognition.lapprox.segments import normalize
from movement_recognition.make_phase_trajectory import to_phase_trajectory

# %%
# Part 0: read and preproc data
df = pd.read_csv(os.getcwd() + '/data/Dormitory_f_158_54_36_accm.csv')  # /tests
df['aT'] = (df['X_value'] ** 2 + df['Y_value'] ** 2 + df['Z_value'] ** 2) ** (1 / 2)
df_sit = df[500: 3100].reset_index(drop=True)
df_mug = df[5000:8100].reset_index(drop=True)
# %% PLOT BLOCK
# a = df['X_value'].plot()
# a2 = df['Y_value'].plot()
# a3 = df['Z_value'].plot()
# a4 = df['aT'].plot()
# plt.legend([a, a2, a3, a4], ['X', 'Y', 'Z', 'aT'])
# df_sit.plot()
# plt.show()
# for i in range(len(fitted)) ]
# print('Ls[0]', Ls[0], f'len Ls: {len(Ls)}')

# column_name='X_value'
# print(df_mug.loc[6050:6100, column_name])
# %% 1. GET SEGMENTS
def get_bounds(df, column_name: str, thr_range: tuple=(-10, 10), min_segment_size:int=500):
    bounds = [normalize.extract_segment_bounds(df=df, column_name=column_name,
                                                min_threshold=i, min_segment_size=min_segment_size)
                    for i in range(thr_range[0], thr_range[1])]

    lens_bounds = [len(b) for b in bounds]

    return bounds[lens_bounds.index(max(lens_bounds))]



# Segments becomes the same length
column_name = 'X_value'
bounds_mug = get_bounds(df_mug, column_name, min_segment_size=800)
bounds_sit = get_bounds(df_sit, column_name, min_segment_size=300)
s_mug = [df_mug.loc[first:second, column_name].values for first, second in bounds_mug]
s_sit = [df_sit.loc[first:second, column_name].values for first, second in bounds_sit]

y_mug = [1]*len(s_mug)
y_sit = [0]*len(s_sit)

# normalized_segments = normalize.normalize_segments(segments)
# segments = [np.array(s) for s in segments]
#%%
# %% 3/ GET PHASE TRAJECTORY

trajectories_mug = [to_phase_trajectory(s, 20) for s in s_mug]
trajectories_sit = [to_phase_trajectory(s, 20) for s in s_sit]
trajectories_mug.extend(trajectories_sit)
y_mug.extend(y_sit)
#MAKE and fit LOCAL MODELS
models = {
    "ssa_5": ssa.Ssa(5),
    "ssa_10": ssa.Ssa(10),
    "ar_2": ar.Ar(2),
    "ar_4": ar.Ar(4),
    "fft_2": fft.Fft(2),
    "fft_5": fft.Fft(5)
}

mapper = joint_mapping.JointMapping(models)
# fitted_trajectories = [mapper.fit(trajectory) for trajectory in s_mug]
fitted_trajectories = [mapper.fit(trajectory) for trajectory in trajectories_mug]
[print(v._params) for k, v in fitted_trajectories[0].models.items()]

#%%
params = [print(v._params) for k, v in fitted_trajectories[0].models.items()]

# [print(v._params.shape) for v in fitted_trajectories[0].models.values()]
# params = [tr._param for tr in fitted_trajectories[0]]
# %% FIT REGRESSION

# Y_matrix = keras.utils.np_utils.to_categorical(Y, len(set(Y)))

# clf = sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='saga', max_iter=1000, C=0.0001)
# linear_model_error = sklearn.model_selection.cross_val_score(clf, X_intermediate, Y, scoring='neg_log_loss', cv=10)
# print(np.mean(linear_model_error))
# clf.fit(params, y_mug)
# Y_pred = clf.predict_proba(params)


# %%SELECT LAM
# grad = np.gradient([np.median(L) for L in Ls])
# print(len(Ls), len(grad))

# print('trajectory[0]', trajectories[0][0])
# print('predicted',res[0].get_predicted()[0])
# [print(f'i={i}', f'j={j}', distance.euclidean(trajectories[0][j], res[i].get_predicted()[j]))  for j in range(1004) for i in range(len(res))]
# 'f1:', sklearn.metrics.f1_score(trajectories[0][0], res[0].get_predicted()[0]))


