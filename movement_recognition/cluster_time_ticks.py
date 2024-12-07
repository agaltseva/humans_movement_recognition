#IVANYCHEV code


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
#
# models = {
#     "ssa_5": ssa.Ssa(5),
#     "ssa_10": ssa.Ssa(10),
#     "ar_2": ar.Ar(2),
#     "ar_4": ar.Ar(4),
#     #     "arma_2_2": arma.Arma(2, 2),
#     "fft_2": fft.Fft(2),
#     "fft_5": fft.Fft(5),
#     "semor_run": semor.Semor(semor_profiles['run']),
#     "semor_walk": semor.Semor(semor_profiles['walk']),
#     "semor_up": semor.Semor(semor_profiles['up']),
#     "semor_down": semor.Semor(semor_profiles['down'])
# }
# mapper = joint_mapping.JointMapping(models)
# X_intermediate = mapper.fit_transform(X)
# mapper.namespace_indices
