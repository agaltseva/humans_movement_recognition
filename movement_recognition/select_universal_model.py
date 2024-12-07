# #𝑍−𝑗  creating
# df_expert_wisdm = pd.read_csv("../data/features/expert_wisdm.csv")
# df_ar_wisdm = pd.read_csv("../data/features/ar_wisdm.csv")
# df_ssa_wisdm = pd.read_csv("../data/features/ssa_wisdm.csv")
# df_fft_wisdm = pd.read_csv("../data/features/fft_wisdm.csv")
# df_wvt_wisdm = pd.read_csv("../data/features/wvt_wisdm.csv")
# df_all_wisdm = pd.read_csv("../data/features/all_wisdm.csv")
#
#
#
# #𝑍−𝑎𝑟
# df__ar_wisdm = pd.concat([df_expert_wisdm,
#                           df_ssa_wisdm.drop('activity',axis=1),
#                           df_fft_wisdm.drop('activity', axis=1),
#                           df_wvt_wisdm.drop('activity', axis=1)], axis = 1, verify_integrity=True)
#
# print(df__ar_wisdm.shape)
# df__ar_wisdm = df__ar_wisdm.T.drop_duplicates().T
# print(df__ar_wisdm.shape)
#
# df__ar_wisdm.to_csv("../data/features/all_ar_wisdm.csv", index=False)
#
#
# #𝑍−𝑠𝑠𝑎
# df__ssa_wisdm = pd.concat([df_expert_wisdm,
#                           df_ar_wisdm.drop('activity', axis=1),
#                           df_fft_wisdm.drop('activity', axis=1),
#                           df_wvt_wisdm.drop('activity', axis=1)], axis = 1, verify_integrity=True)
# print(df__ssa_wisdm.shape)
# df__ssa_wisdm = df__ssa_wisdm.T.drop_duplicates().T
# print(df__ssa_wisdm.shape)
#
# df__ssa_wisdm.to_csv("../data/features/all_ssa_wisdm.csv", index=False)
#
#
# #𝑍−𝑓𝑓𝑡
# df__fft_wisdm = pd.concat([df_expert_wisdm,
#                           df_ar_wisdm.drop('activity', axis=1),
#                           df_ssa_wisdm.drop('activity',axis=1),
#                           df_wvt_wisdm.drop('activity', axis=1)], axis = 1, verify_integrity=True)
# print(df__fft_wisdm.shape)
# df__fft_wisdm = df__fft_wisdm.T.drop_duplicates().T
# print(df__fft_wisdm.shape)
# ​
# df__fft_wisdm.to_csv("../data/features/all_fft_wisdm.csv", index=False)
#
#
# #𝑍−𝑒𝑥𝑝𝑒𝑟𝑡
# df__expert_wisdm = pd.concat([df_fft_wisdm,
#                           df_ar_wisdm.drop('activity', axis=1),
#                           df_ssa_wisdm.drop('activity',axis=1),
#                           df_wvt_wisdm.drop('activity', axis=1)], axis = 1, verify_integrity=True)
# print(df__expert_wisdm.shape)
# df__expert_wisdm = df__expert_wisdm.T.drop_duplicates().T
# print(df__expert_wisdm.shape)
#
# df__expert_wisdm.to_csv("../data/features/all_expert_wisdm.csv", index=False)
#
#
# #𝑍−𝑤𝑣𝑡
# df__wvt_wisdm = pd.concat([df_fft_wisdm,
#                           df_ar_wisdm.drop('activity', axis=1),
#                           df_ssa_wisdm.drop('activity',axis=1),
#                           df_expert_wisdm.drop('activity', axis=1)], axis = 1, verify_integrity=True)
# print(df__wvt_wisdm.shape)
# df__wvt_wisdm = df__wvt_wisdm.T.drop_duplicates().T
# print(df__wvt_wisdm.shape)
#
# df__wvt_wisdm.to_csv("../data/features/all_wvt_wisdm.csv", index=False)



#IVANYCHEV code
# def transform_to_inputs(Y, n_classes):
#     Y_binary = np.zeros((Y.shape[0], n_classes))
#     for i in range(n_classes):
#         Y_binary[:, i] = (Y == i)
#
#     return Y_binary
#
#
# def test_neural_network(X, Y, units, cv=10):
#     n_classes = len(set(Y))
#     kfold = sklearn.model_selection.StratifiedKFold(n_splits=cv, random_state=1)
#
#     scores = []
#     for train_index, test_index in kfold.split(X, Y):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = (keras.utils.np_utils.to_categorical(Y[train_index], n_classes),
#                            keras.utils.np_utils.to_categorical(Y[test_index], n_classes))
#         model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(units,), activation='logistic')
#
#         #         model = keras.models.Sequential()
#         #         model.add(keras.layers.Dense(units, activation='relu', input_dim=X_train.shape[1]))
#         #         model.add(keras.layers.Dense(n_classes, activation='sigmoid'))
#         #         model.compile(loss='categorical_crossentropy',
#         #                       optimizer='rmsprop')
#         model.fit(X_train, Y[train_index])
#         y_pred = model.predict_proba(X_test)
#         scores.append(sklearn.metrics.log_loss(Y[test_index], y_pred))
#     #         scores.append(model.evaluate(X_test, y_test, verbose=0)
#
#     return np.array(scores)