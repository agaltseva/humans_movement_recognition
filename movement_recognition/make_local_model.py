# import pandas as pd
#
# def get_autoregressive_names(params):
#     n = params[0]
#     feature_names = []
#     for ax in ['x', 'y', 'z']:
#         feature_names += ['intercept_' + ax]
#         for i in range(n):
#             feature_names += ['coef_' + str(i) + '_' + ax]
#
#     return feature_names
#
#
# def get_autoregressive_features(ts, params):
#     n = params[0]
#     x = ts[0]
#     y = ts[1]
#     z = ts[2]
#     m = x.shape[0]
#     features = []
#     X = np.zeros([m - n, n])
#     Y = np.zeros(m - n)
#     for axis in [x, y, z]:
#         for i in range(m - n):
#             X[i, :] = axis[i:i + n]
#             Y[i] = axis[i + n]
#         lr = LinearRegression()
#         lr.fit(X, Y)
#         features.append(lr.intercept_)
#         features.extend(lr.coef_)
#
#     return features
#
#  # Spectrum analysis
# def get_spectrum_names(params):
#     n = params[0]
#     feature_names = []
#     for ax in ['x', 'y', 'z']:
#         for i in range(n):
#             feature_names += ['eigv_' + str(i) + '_' + ax]
#
#     return feature_names
#
#
# def get_spectrum_features(ts, params):
#     n = params[0]
#     x = ts[0]
#     y = ts[1]
#     z = ts[2]
#     m = x.shape[0]
#     features = []
#     X = np.zeros([m - n, n])
#     Y = np.zeros(m - n)
#     for axis in [x, y, z]:
#         for i in range(m - n):
#             X[i, :] = axis[i:i + n]
#         h = sc.linalg.svd(X.T.dot(X), compute_uv=False, overwrite_a=True)
#         features.extend(h)
#
#     return features
#
# ## Fast Fourier Transform
# def get_fft_names(params):
#     n = params[0]
#     feature_names = []
#     for ax in ['x', 'y', 'z']:
#         for i in range(2 * n):
#             feature_names += ['fft_coef_' + str(i) + '_' + ax]
#
#     return feature_names
#
#
# def get_fft_features(ts, params):
#     n = params[0]
#     x = ts[0]
#     y = ts[1]
#     z = ts[2]
#     m = x.shape[0]
#     features = []
#     # X = np.zeros([m-n, n])
#     # Y = np.zeros(m-n)
#     for axis in [x, y, z]:
#         h = sc.fftpack.fft(axis, n, axis=-1, overwrite_x=False)
#         features.extend(polar(h))
#
#     return list(np.array(features) ** 2)
#
#
# def polar(lis):
#     c = []
#     for i in range(len(lis)):
#         c.extend(cmath.polar(lis[i]))
#
#     return c
#
# # Wavelet transform
# def get_wvt_names(params):
#     n = params[0]
#     feature_names = []
#     for ax in ['x', 'y', 'z']:
#         for i in range(2 * n):
#             feature_names += ['wvt_coef_' + str(i) + '_' + ax]
#
#     return feature_names
#
#
# def get_wvt_features(ts, params):
#     n = params[0]
#     x = ts[0]
#     y = ts[1]
#     z = ts[2]
#     m = x.shape[0]
#     features = []
#     for axis in [x, y, z]:
#         h = pywt.dwt(axis, 'db1')
#         features.extend(h[0][:n])
#         features.extend(h[1][:n])
#
#     return features
#
# ## All features
# df_all_wisdm = pd.concat([df_expert_wisdm,
#                           df_ar_wisdm.drop('activity', axis=1),
#                           df_ssa_wisdm.drop('activity',axis=1),
#                           df_fft_wisdm.drop('activity', axis=1),
#                           df_wvt_wisdm.drop('activity', axis=1)], axis = 1, verify_integrity=True)
# df_all_wisdm = df_all_wisdm.T.drop_duplicates().

#IVANYCHEV module 'models'