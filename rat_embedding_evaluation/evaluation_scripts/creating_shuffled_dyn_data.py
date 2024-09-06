import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# creating shuffled data
for rat_name in ['achilles', 'gatsby', 'cicero', 'buddy']:
    """
    algorithm = 'BunDLeNet'
    print(algorithm, ' rat_name: ', rat_name)

    file_pattern = f'data/generated/saved_Y/{{}}__{algorithm}_rat_{rat_name}'
    y0_tr = np.loadtxt(file_pattern.format('y0_tr'))
    y1_tr = np.loadtxt(file_pattern.format('y1_tr'))
    y0_tst = np.loadtxt(file_pattern.format('y0_tst'))
    y1_tst = np.loadtxt(file_pattern.format('y1_tst'))
    b_train_1 = np.loadtxt(file_pattern.format('b_train_1'))
    b_test_1 = np.loadtxt(file_pattern.format('b_test_1'))

    print(y0_tr.shape, y1_tr.shape, y0_tst.shape, y1_tst.shape)
    print(b_train_1.shape, b_test_1.shape)

    y0_tr_sh = np.random.permutation(y0_tr)
    y1_tr_sh = np.random.permutation(y1_tr)
    y0_tst_sh = np.random.permutation(y0_tst)
    y1_tst_sh = np.random.permutation(y1_tst)

    algorithm = 'y_shuffled_BunDLeNet'
    file_pattern = f'data/generated/saved_Y/{{}}__{algorithm}_rat_{rat_name}'
    np.savetxt(file_pattern.format('y0_tr'), y0_tr_sh)
    np.savetxt(file_pattern.format('y1_tr'), y1_tr_sh)
    np.savetxt(file_pattern.format('y0_tst'), y0_tst_sh)
    np.savetxt(file_pattern.format('y1_tst'), y1_tst_sh)
    np.savetxt(file_pattern.format('b_train_1'), b_train_1)
    np.savetxt(file_pattern.format('b_test_1'), b_test_1)

    # creating point embedding
    y0_tr_point = np.ones_like(y0_tr) + 0.001*np.random.rand(y0_tr.shape[0], y0_tr.shape[1])
    y1_tr_point = np.ones_like(y1_tr) + 0.001*np.random.rand(y1_tr.shape[0], y1_tr.shape[1])
    y0_tst_point = np.ones_like(y0_tst) + 0.001*np.random.rand(y0_tst.shape[0], y0_tst.shape[1])
    y1_tst_point = np.ones_like(y1_tst) + 0.001*np.random.rand(y1_tst.shape[0], y1_tst.shape[1])

    algorithm = 'point_embedding_noisy'
    file_pattern = f'data/generated/saved_Y/{{}}__{algorithm}_rat_{rat_name}'
    np.savetxt(file_pattern.format('y0_tr'), y0_tr_point)
    np.savetxt(file_pattern.format('y1_tr'), y1_tr_point)
    np.savetxt(file_pattern.format('y0_tst'), y0_tst_point)
    np.savetxt(file_pattern.format('y1_tst'), y1_tst_point)
    np.savetxt(file_pattern.format('b_train_1'), b_train_1)
    np.savetxt(file_pattern.format('b_test_1'), b_test_1)
    """
    algorithm = 'linear_dynamics'
    def linear_dynamics_generator(sequence_length):
        theta = 0.01
        T_Y = np.array([[np.cos(theta)*np.cos(3*theta), - np.sin(theta), - np.sin(3*theta)],
                      [np.sin(theta), np.cos(theta) , 0],
                      [np.sin(3*theta), 0, np.cos(3*theta)]])
        y_0 = np.array([1, -1, 0.5])
        y = [y_0]
        for i in range(sequence_length):
            y.append( T_Y @ y[-1] )
        return np.array(y)


    y0_tr = np.zeros((3232,3))
    y1_tr = np.zeros((3232,3))
    y0_tst = np.zeros((3232,3))
    y1_tst = np.zeros((3232,3))

    y0_tr_linear = linear_dynamics_generator(y0_tr.shape[0])
    y1_tr_linear = linear_dynamics_generator(y1_tr.shape[0])
    y0_tst_linear = linear_dynamics_generator(y0_tst.shape[0])
    y1_tst_linear = linear_dynamics_generator(y1_tst.shape[0])

    file_pattern = f'data/generated/saved_Y/{{}}__{algorithm}_rat_{rat_name}'
    np.savetxt(file_pattern.format('y0_tr'), y0_tr_linear)
    np.savetxt(file_pattern.format('y1_tr'), y1_tr_linear)
    np.savetxt(file_pattern.format('y0_tst'), y0_tst_linear)
    np.savetxt(file_pattern.format('y1_tst'), y1_tst_linear)
    np.savetxt(file_pattern.format('b_train_1'), b_train_1)
    np.savetxt(file_pattern.format('b_test_1'), b_test_1)

