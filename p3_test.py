from sklearn import preprocessing
from s1_utils import *
import os
import scipy.io as sio
from lib import models, graph, coarsening, utils
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import shapiro, spearmanr
from statsmodels.stats.multitest import fdrcorrection

file_path = "/Users/qiner/OneDrive/OneDrive - McGill University/McGill study/urgency/data/fMRI/2-BN246_atlas/"
label_path = "/Users/qiner/OneDrive/OneDrive - McGill University/McGill study/urgency/data/fMRI/1-TR_label/"
behav_path = "/Users/qiner/OneDrive/OneDrive - McGill University/McGill study/urgency/data/fMRI/0-raw_behav/"

TR = 0.719
seq_len = 8  # because each trial has 8-18 TRs, so each trial extract 8 minimal TRs,
# average is 12 TRs (mean RT = 2.39, whole trial last around 12 TRs)

fmri_data = []
trial_resp_label = []
trial_rt = []
for f in list_files(file_path, "_subregions_data.mat"):  # 9002_session1_morph1
    # fmri subregions data
    subreg_data = sio.loadmat(file_path + f)['subregions_data']
    subreg_data = preprocessing.scale(subreg_data, axis=1)  # normalize the data

    # fmri behavior data
    TR_trial_number = sio.loadmat(label_path + f[:-20] + '_TR_label.mat')['TR_trial_number']
    TR_trial_number = np.squeeze(TR_trial_number)
    # TR label data
    TR_task_stage = sio.loadmat(label_path + f[:-20] + '_TR_label.mat')['TR_task_stage']
    TR_task_stage = np.squeeze(TR_task_stage)
    # task behav data
    behav_file = list_files(behav_path, f[:5] + f[19:21])
    task_info = sio.loadmat(behav_path + behav_file[0])['task_info']

    if len(trial_resp_label) == 0:
        # response
        trial_resp_label = np.int8(task_info[:, 5])
        # response time
        trial_rt = task_info[:, 12] - task_info[:, 11]
        TR_ind = np.where(TR_task_stage == 3.5)[0]  # response stage
        # preprocessing fmri data
        fmri_data = subreg_data[:, TR_ind].T
    else:
        # response
        trial_resp_label = np.concatenate((trial_resp_label, np.int8(task_info[:, 5])))
        # response time
        trial_rt = np.concatenate((trial_rt, task_info[:, 12] - task_info[:, 11]))
        TR_ind = np.where(TR_task_stage == 3.5)[0]  # response stage
        # preprocessing fmri data
        fmri_data = np.vstack((fmri_data, subreg_data[:, TR_ind].T))

    # # split data according to trial
    # trial_num = np.unique(TR_trial_number)[1:]  # except trial ID -1
    # for tri in range(trial_num.size):
    #     TR_ind = np.where(TR_trial_number == tri)[0]
    #     TR_ind = TR_ind[-seq_len:]
    #     # preprocessing fmri data
    #     fmri_data.append(subreg_data[:, TR_ind].T)

X = fmri_data  # netw_data  # samples*features
Y = trial_resp_label
noResInd = np.where(Y == 0)[0]
X = np.delete(X, noResInd, 0)
Y = np.delete(Y, noResInd)
Y[np.where(Y == 1)] = 0
Y[np.where(Y == 3)] = 1

n, d = X.shape  # Number of samples. # Dimensionality.

# # correlation adjacent
# p_thres = 0.05
# adj, _ = spearmanr(X, axis=0)  # correlation network
# # a, b = fdrcorrection(p_val, p_thres, method='negcorr')
# # adj[p_val < p_thres / (X.shape[1] * X.shape[1])] = 0

# structural connectivity
adj = sio.loadmat('BNA_matrix_binary.mat')['BNA_matrix_binary']
# functional connectivity, can't transfer the dimension 91*109*91*246??

A = scipy.sparse.coo_matrix(np.abs(np.float32(adj)))
print('d = |V| = {}, k|V| < |E| = {}'.format(d, A.nnz))
graphs, perm = coarsening.coarsen(A, levels=3, self_connections=False)
L = [graph.laplacian(A, normalized=True) for A in graphs]
# graph.plot_spectrum(L)

n_train = np.int64(n * 0.8)  # np.int(n * 0.8)  # n // 2  #
n_val = n // 10

X_train = X[:n_train]
X_val = X[n_train:n_train + n_val]
X_test = X[n_train + n_val:]
X_train = coarsening.perm_data(X_train, perm)
X_val = coarsening.perm_data(X_val, perm)
X_test = coarsening.perm_data(X_test, perm)

y_train = Y[:n_train]
y_val = Y[n_train:n_train + n_val]
y_test = Y[n_train + n_val:]

# Number of classes.
C = np.unique(Y).size
# C = np.unique(TR_task_stage).size
params = dict()
params['dir_name'] = 'demo'
params['num_epochs'] = 40
params['batch_size'] = 100
params['eval_frequency'] = 200
# Building blocks.
params['filter'] = 'chebyshev5'
params['brelu'] = 'b1relu'
params['pool'] = 'apool1'
# Architecture.
# F can't set be 64, only 32
params['F'] = [32, 32]  # Number of graph convolutional filters.
params['p'] = [4, 2]  # Pooling sizes.
params['K'] = [10, 10]  # Polynomial orders.
params['M'] = [128, C]  # Output dimensionality of fully connected layers.
# Optimization.
params['regularization'] = 5e-4
params['dropout'] = 0.5
params['learning_rate'] = 1e-3
params['decay_rate'] = 0.95
params['momentum'] = 0.9
params['decay_steps'] = n_train / params['batch_size']

model = models.cgcnn(L, **params)
accuracy, loss, t_step = model.fit(X_train, y_train, X_val, y_val)

fig, ax1 = plt.subplots(figsize=(15, 5))
ax1.plot(accuracy, 'b.-')
ax1.set_ylabel('validation accuracy', color='b')
ax2 = ax1.twinx()
ax2.plot(loss, 'g.-')
ax2.set_ylabel('training loss', color='g')
plt.show()

print('Time per step: {:.2f} ms'.format(t_step * 1000))

res = model.evaluate(X_test, y_test)
print(res[0])
print(res[0])
