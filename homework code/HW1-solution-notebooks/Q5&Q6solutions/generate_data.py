import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine

dataset_path = 'datasets'

### generate synthesized data ###
def generate_separable_for_perceptron():
    N = 200
    pos = 0.4
    labels = np.random.binomial(1, pos, size=N)

    # define the boundary
    # Ax + By - C = 0
    A = 1
    B = 1
    C = 0

    # define the margin
    m = 0.1

    # defines xs
    offset = np.sqrt(A**2 + B**2) * m
    bound_1 = offset - C
    bound_2 = -offset - C
    positive_x1 = np.random.normal(
        loc=0.3, scale=0.2, size=N)
    negative_x1 = np.random.normal(
        loc=-0.4, scale=0.2, size=N)
    x1 = labels * positive_x1 + (1-labels) * negative_x1
    positive_x2 = np.random.uniform(
        low=(bound_1-A*x1)/B, high=0.6)
    negative_x2 = np.random.uniform(
        low=-0.7, high=(bound_2-A*x1)/B)
    x2 = labels * positive_x2 + (1 - labels) * negative_x2
    xs = np.vstack((x1, x2)).transpose()
    labels = labels * 2 - 1

    output_path = os.path.join(dataset_path, 'q1_separable.npz')
    np.savez(output_path, x=xs, y=labels)

### generate synthesized data ###
def generate_nonseparable_for_perceptron():
    N = 200
    pos = 0.4
    labels = np.random.binomial(1, pos, size=N)
    
    # the last few points are wrong
    err_rate = 0.05
    N_err = int(N * err_rate)
    mask = labels.copy()
    err_idxs = np.arange(N)
    np.random.shuffle(err_idxs)
    err_idxs = err_idxs[:N_err]
    mask[err_idxs] = 1 - mask[err_idxs]

    # define the boundary
    # Ax + By - C = 0
    A = 1
    B = 1
    C = 0

    # define the margin
    m = 0.02

    # defines xs
    offset = np.sqrt(A**2 + B**2) * m
    bound_1 = offset - C
    bound_2 = -offset - C
    positive_x1 = np.random.normal(
        loc=0.3, scale=0.2, size=N)
    negative_x1 = np.random.normal(
        loc=-0.4, scale=0.2, size=N)
    x1 = mask * positive_x1 + (1-mask) * negative_x1
    positive_x2 = np.random.uniform(
        low=(bound_1-A*x1)/B, high=0.6)
    negative_x2 = np.random.uniform(
        low=-0.7, high=(bound_2-A*x1)/B)
    x2 = mask * positive_x2 + (1 - mask) * negative_x2
    xs = np.vstack((x1, x2)).transpose()
    labels = labels * 2 - 1

    output_path = os.path.join(dataset_path, 'q1_non_separable.npz')
    np.savez(output_path, x=xs, y=labels)

def generate_separable_for_SVM():
    N = 200
    pos = 0.45
    labels = np.random.binomial(1, pos, size=N)

    # define the boundary
    # Ax + By - C = 0
    A = 1
    B = 1
    C1 = -0.8
    C2 = -0.4

    # defines xs
    positive_x1 = np.random.normal(
        loc=0.5, scale=0.2, size=N)
    negative_x1 = np.random.normal(
        loc=-0.1, scale=0.2, size=N)
    x1 = labels * positive_x1 + (1-labels) * negative_x1
    positive_x2 = np.random.uniform(
        low=(-C1-A*x1)/B, high=1.8)
    negative_x2 = np.random.uniform(
        low=-0.2, high=(-C2-A*x1)/B)
    x2 = labels * positive_x2 + (1 - labels) * negative_x2
    xs = np.vstack((x1, x2)).transpose()
    labels = labels * 2 - 1

    output_path = os.path.join(dataset_path, 'q2_separable.npz')
    np.savez(output_path, x=xs, y=labels)

def generate_nonseparable_for_SVM():
    pos = 0.45 # positive rate

    # generate well-classified samples
    N = 200
    pos = 0.45
    labels = np.random.binomial(1, pos, size=N)
    N_good = 175
    N_margin_errors = 17
    N_class_errors = 8

    # define the boundary
    # Ax + By - C = 0
    A = 1
    B = 1
    C1 = -0.8
    C2 = -0.4

    # defines xs
    positive_x1 = np.random.normal(
        loc=0.5, scale=0.2, size=N)
    negative_x1 = np.random.normal(
        loc=-0.1, scale=0.2, size=N)
    x1 = labels * positive_x1 + (1-labels) * negative_x1
    positive_x2 = np.random.uniform(
        low=(-C1-A*x1)/B, high=1.8)
    positive_x2[N_good:-N_class_errors] = np.random.uniform(
        low=(-(C1+C2)/2-A*x1[N_good:-N_class_errors])/B, 
        high=(-C1-A*x1[N_good:-N_class_errors])/B)
    positive_x2[-N_class_errors:] = np.random.uniform(
        low=-0.2, high=(-(C1+C2)/2-A*x1[-N_class_errors:])/B)
    negative_x2 = np.random.uniform(
        low=-0.2, high=(-C2-A*x1)/B)
    negative_x2[N_good:-N_class_errors] = np.random.uniform(
        low=(-C2-A*x1[N_good:-N_class_errors])/B, 
        high=(-(C1+C2)/2-A*x1[N_good:-N_class_errors])/B)
    negative_x2[-N_class_errors:] = np.random.uniform(
        low=(-(C1+C2)/2-A*x1[-N_class_errors:])/B, high=1.8)
    x2 = labels * positive_x2 + (1 - labels) * negative_x2
    xs = np.vstack((x1, x2)).transpose()
    labels = labels * 2 - 1

    output_path = os.path.join(dataset_path, 'q2_non_separable.npz')
    np.savez(output_path, x=xs, y=labels)

def get_wine_data():
    wine_data, wine_target = load_wine(return_X_y=True)
    output_path = os.path.join(dataset_path, 'wine.npz')
    filtered_idxs = np.where((wine_target==0) | (wine_target==1))[0]
    wine_data = wine_data[filtered_idxs]
    wine_target = wine_target[filtered_idxs]
    wine_target = wine_target * 2 - 1
    np.savez(output_path, x=wine_data, y=wine_target)
