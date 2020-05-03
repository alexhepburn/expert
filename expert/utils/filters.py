"""
The :mod:`expert.utils.filters` module holds filters used in the
:mod:`exeprt.models.pyramids` module.
"""
# Author: Alex Hepburn <alex.hepburn@bristol.ac.uk>
# License: new BSD

import numpy as np

import torch

STEERABLE_SPATIAL_FILTERS = {
    'lo0filt': torch.from_numpy(np.array([
        [-8.701000e-05, -1.354280e-03, -1.601260e-03, -5.033700e-04,
        2.524010e-03, -5.033700e-04, -1.601260e-03, -1.354280e-03,
        -8.701000e-05],
        [-1.354280e-03, 2.921580e-03, 7.522720e-03, 8.224420e-03, 1.107620e-03,
        8.224420e-03, 7.522720e-03, 2.921580e-03, -1.354280e-03],
        [-1.601260e-03, 7.522720e-03, -7.061290e-03, -3.769487e-02,
        -3.297137e-02, -3.769487e-02, -7.061290e-03, 7.522720e-03,
        -1.601260e-03],
        [-5.033700e-04, 8.224420e-03, -3.769487e-02, 4.381320e-02, 1.811603e-01,
        4.381320e-02, -3.769487e-02, 8.224420e-03, -5.033700e-04],
        [2.524010e-03, 1.107620e-03, -3.297137e-02, 1.811603e-01, 4.376250e-01,
        1.811603e-01, -3.297137e-02, 1.107620e-03, 2.524010e-03],
        [-5.033700e-04, 8.224420e-03, -3.769487e-02, 4.381320e-02, 1.811603e-01,
        4.381320e-02, -3.769487e-02, 8.224420e-03, -5.033700e-04],
        [-1.601260e-03, 7.522720e-03, -7.061290e-03, -3.769487e-02,
        -3.297137e-02, -3.769487e-02, -7.061290e-03, 7.522720e-03,
        -1.601260e-03],
        [-1.354280e-03, 2.921580e-03, 7.522720e-03, 8.224420e-03, 1.107620e-03,
        8.224420e-03, 7.522720e-03, 2.921580e-03, -1.354280e-03],
        [-8.701000e-05, -1.354280e-03, -1.601260e-03, -5.033700e-04,
        2.524010e-03, -5.033700e-04, -1.601260e-03, -1.354280e-03,
        -8.701000e-05]], dtype=np.float32)).reshape((1, 1, 9, 9)),
    'hi0filt': torch.from_numpy(np.array([
        [-9.570000e-04, -2.424100e-04, -1.424720e-03, -8.742600e-04,
        -1.166810e-03, -8.742600e-04, -1.424720e-03, -2.424100e-04,
        -9.570000e-04],
        [-2.424100e-04, -4.317530e-03, 8.998600e-04, 9.156420e-03, 1.098012e-02,
        9.156420e-03, 8.998600e-04, -4.317530e-03, -2.424100e-04],
        [-1.424720e-03, 8.998600e-04, 1.706347e-02, 1.094866e-02, -5.897780e-03,
        1.094866e-02, 1.706347e-02, 8.998600e-04, -1.424720e-03],
        [-8.742600e-04, 9.156420e-03, 1.094866e-02, -7.841370e-02,
        -1.562827e-01, -7.841370e-02, 1.094866e-02, 9.156420e-03,
        -8.742600e-04],
        [-1.166810e-03, 1.098012e-02, -5.897780e-03, -1.562827e-01,
        7.282593e-01, -1.562827e-01, -5.897780e-03, 1.098012e-02,
        -1.166810e-03],
        [-8.742600e-04, 9.156420e-03, 1.094866e-02, -7.841370e-02,
        -1.562827e-01, -7.841370e-02, 1.094866e-02, 9.156420e-03,
        -8.742600e-04],
        [-1.424720e-03, 8.998600e-04, 1.706347e-02, 1.094866e-02, -5.897780e-03,
        1.094866e-02, 1.706347e-02, 8.998600e-04, -1.424720e-03],
        [-2.424100e-04, -4.317530e-03, 8.998600e-04, 9.156420e-03, 1.098012e-02,
        9.156420e-03, 8.998600e-04, -4.317530e-03, -2.424100e-04],
        [-9.570000e-04, -2.424100e-04, -1.424720e-03, -8.742600e-04,
        -1.166810e-03, -8.742600e-04, -1.424720e-03, -2.424100e-04,
        -9.570000e-04]], dtype=np.float32)).reshape((1, 1, 9, 9)),
    'lofilt': torch.from_numpy(np.array([
    [-4.350000e-05, 1.207800e-04, -6.771400e-04, -1.243400e-04, -8.006400e-04,
    -1.597040e-03, -2.516800e-04, -4.202000e-04, 1.262000e-03, -4.202000e-04,
    -2.516800e-04, -1.597040e-03, -8.006400e-04, -1.243400e-04, -6.771400e-04,
    1.207800e-04, -4.350000e-05],
    [1.207800e-04, 4.460600e-04, -5.814600e-04, 5.621600e-04, -1.368800e-04,
    2.325540e-03, 2.889860e-03, 4.287280e-03, 5.589400e-03, 4.287280e-03,
    2.889860e-03, 2.325540e-03, -1.368800e-04, 5.621600e-04, -5.814600e-04,
    4.460600e-04, 1.207800e-04],
    [-6.771400e-04, -5.814600e-04, 1.460780e-03, 2.160540e-03, 3.761360e-03,
    3.080980e-03, 4.112200e-03, 2.221220e-03, 5.538200e-04, 2.221220e-03,
    4.112200e-03, 3.080980e-03, 3.761360e-03, 2.160540e-03, 1.460780e-03,
    -5.814600e-04, -6.771400e-04],
    [-1.243400e-04, 5.621600e-04, 2.160540e-03, 3.175780e-03, 3.184680e-03,
    -1.777480e-03, -7.431700e-03, -9.056920e-03, -9.637220e-03, -9.056920e-03,
    -7.431700e-03, -1.777480e-03, 3.184680e-03, 3.175780e-03, 2.160540e-03,
    5.621600e-04, -1.243400e-04],
    [-8.006400e-04, -1.368800e-04, 3.761360e-03, 3.184680e-03, -3.530640e-03,
    -1.260420e-02, -1.884744e-02, -1.750818e-02, -1.648568e-02, -1.750818e-02,
    -1.884744e-02, -1.260420e-02, -3.530640e-03, 3.184680e-03, 3.761360e-03,
    -1.368800e-04, -8.006400e-04],
    [-1.597040e-03, 2.325540e-03, 3.080980e-03, -1.777480e-03, -1.260420e-02,
    -2.022938e-02, -1.109170e-02, 3.955660e-03, 1.438512e-02, 3.955660e-03,
    -1.109170e-02, -2.022938e-02, -1.260420e-02, -1.777480e-03, 3.080980e-03,
    2.325540e-03, -1.597040e-03],
    [-2.516800e-04, 2.889860e-03, 4.112200e-03, -7.431700e-03, -1.884744e-02,
    -1.109170e-02, 2.190660e-02, 6.806584e-02, 9.058014e-02, 6.806584e-02,
    2.190660e-02, -1.109170e-02, -1.884744e-02, -7.431700e-03, 4.112200e-03,
    2.889860e-03, -2.516800e-04],
    [-4.202000e-04, 4.287280e-03, 2.221220e-03, -9.056920e-03, -1.750818e-02,
    3.955660e-03, 6.806584e-02, 1.445500e-01, 1.773651e-01, 1.445500e-01,
    6.806584e-02, 3.955660e-03, -1.750818e-02, -9.056920e-03, 2.221220e-03,
    4.287280e-03, -4.202000e-04],
    [1.262000e-03, 5.589400e-03, 5.538200e-04, -9.637220e-03, -1.648568e-02,
    1.438512e-02, 9.058014e-02, 1.773651e-01, 2.120374e-01, 1.773651e-01,
    9.058014e-02, 1.438512e-02, -1.648568e-02, -9.637220e-03, 5.538200e-04,
    5.589400e-03, 1.262000e-03],
    [-4.202000e-04, 4.287280e-03, 2.221220e-03, -9.056920e-03, -1.750818e-02,
    3.955660e-03, 6.806584e-02, 1.445500e-01, 1.773651e-01, 1.445500e-01,
    6.806584e-02, 3.955660e-03, -1.750818e-02, -9.056920e-03, 2.221220e-03,
    4.287280e-03, -4.202000e-04],
    [-2.516800e-04, 2.889860e-03, 4.112200e-03, -7.431700e-03, -1.884744e-02,
    -1.109170e-02, 2.190660e-02, 6.806584e-02, 9.058014e-02, 6.806584e-02,
    2.190660e-02, -1.109170e-02, -1.884744e-02, -7.431700e-03, 4.112200e-03,
    2.889860e-03, -2.516800e-04],
    [-1.597040e-03, 2.325540e-03, 3.080980e-03, -1.777480e-03, -1.260420e-02,
    -2.022938e-02, -1.109170e-02, 3.955660e-03, 1.438512e-02, 3.955660e-03,
    -1.109170e-02, -2.022938e-02, -1.260420e-02, -1.777480e-03, 3.080980e-03,
    2.325540e-03, -1.597040e-03],
    [-8.006400e-04, -1.368800e-04, 3.761360e-03, 3.184680e-03, -3.530640e-03,
    -1.260420e-02, -1.884744e-02, -1.750818e-02, -1.648568e-02, -1.750818e-02,
    -1.884744e-02, -1.260420e-02, -3.530640e-03, 3.184680e-03, 3.761360e-03,
    -1.368800e-04, -8.006400e-04],
    [-1.243400e-04, 5.621600e-04, 2.160540e-03, 3.175780e-03, 3.184680e-03,
    -1.777480e-03, -7.431700e-03, -9.056920e-03, -9.637220e-03, -9.056920e-03,
    -7.431700e-03, -1.777480e-03, 3.184680e-03, 3.175780e-03, 2.160540e-03,
    5.621600e-04, -1.243400e-04],
    [-6.771400e-04, -5.814600e-04, 1.460780e-03, 2.160540e-03, 3.761360e-03,
    3.080980e-03, 4.112200e-03, 2.221220e-03, 5.538200e-04, 2.221220e-03,
    4.112200e-03, 3.080980e-03, 3.761360e-03, 2.160540e-03, 1.460780e-03,
    -5.814600e-04, -6.771400e-04],
    [1.207800e-04, 4.460600e-04, -5.814600e-04, 5.621600e-04, -1.368800e-04,
    2.325540e-03, 2.889860e-03, 4.287280e-03, 5.589400e-03, 4.287280e-03,
    2.889860e-03, 2.325540e-03, -1.368800e-04, 5.621600e-04, -5.814600e-04,
    4.460600e-04, 1.207800e-04],
    [-4.350000e-05, 1.207800e-04, -6.771400e-04, -1.243400e-04, -8.006400e-04,
    -1.597040e-03, -2.516800e-04, -4.202000e-04, 1.262000e-03, -4.202000e-04,
    -2.516800e-04, -1.597040e-03, -8.006400e-04, -1.243400e-04, -6.771400e-04,
    1.207800e-04, -4.350000e-05]], dtype=np.float32)).reshape((1, 1, 17, 17)),
    'bfilts': torch.from_numpy(np.array([
    [[6.125880e-03, -8.052600e-03, -2.103714e-02, -1.536890e-02, -1.851466e-02,
    -1.536890e-02, -2.103714e-02, -8.052600e-03, 6.125880e-03],
     [-1.287416e-02, -9.611520e-03, 1.023569e-02, 6.009450e-03, 1.872620e-03,
     6.009450e-03, 1.023569e-02, -9.611520e-03, -1.287416e-02],
     [-5.641530e-03, 4.168400e-03, -2.382180e-02, -5.375324e-02, -2.076086e-02,
     -5.375324e-02, -2.382180e-02, 4.168400e-03, -5.641530e-03],
     [-8.957260e-03, -1.751170e-03, -1.836909e-02, 1.265655e-01, 2.996168e-01,
     1.265655e-01, -1.836909e-02, -1.751170e-03, -8.957260e-03],
     [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
     0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
     [8.957260e-03, 1.751170e-03, 1.836909e-02, -1.265655e-01, -2.996168e-01,
     -1.265655e-01, 1.836909e-02, 1.751170e-03, 8.957260e-03],
     [5.641530e-03, -4.168400e-03, 2.382180e-02, 5.375324e-02, 2.076086e-02,
     5.375324e-02, 2.382180e-02, -4.168400e-03, 5.641530e-03],
     [1.287416e-02, 9.611520e-03, -1.023569e-02, -6.009450e-03, -1.872620e-03,
     -6.009450e-03, -1.023569e-02, 9.611520e-03, 1.287416e-02],
     [-6.125880e-03, 8.052600e-03, 2.103714e-02, 1.536890e-02, 1.851466e-02,
     1.536890e-02, 2.103714e-02, 8.052600e-03, -6.125880e-03]],
    [[-6.125880e-03, 1.287416e-02, 5.641530e-03, 8.957260e-03, 0.000000e+00,
    -8.957260e-03, -5.641530e-03, -1.287416e-02, 6.125880e-03],
     [8.052600e-03, 9.611520e-03, -4.168400e-03, 1.751170e-03, 0.000000e+00,
     -1.751170e-03, 4.168400e-03, -9.611520e-03, -8.052600e-03],
     [2.103714e-02, -1.023569e-02, 2.382180e-02, 1.836909e-02, 0.000000e+00,
     -1.836909e-02, -2.382180e-02, 1.023569e-02, -2.103714e-02],
     [1.536890e-02, -6.009450e-03, 5.375324e-02, -1.265655e-01, 0.000000e+00,
     1.265655e-01, -5.375324e-02, 6.009450e-03, -1.536890e-02],
     [1.851466e-02, -1.872620e-03, 2.076086e-02, -2.996168e-01, 0.000000e+00,
     2.996168e-01, -2.076086e-02, 1.872620e-03, -1.851466e-02],
     [1.536890e-02, -6.009450e-03, 5.375324e-02, -1.265655e-01, 0.000000e+00,
     1.265655e-01, -5.375324e-02, 6.009450e-03, -1.536890e-02],
     [2.103714e-02, -1.023569e-02, 2.382180e-02, 1.836909e-02, 0.000000e+00,
     -1.836909e-02, -2.382180e-02, 1.023569e-02, -2.103714e-02],
     [8.052600e-03, 9.611520e-03, -4.168400e-03, 1.751170e-03, 0.000000e+00,
     -1.751170e-03, 4.168400e-03, -9.611520e-03, -8.052600e-03],
     [-6.125880e-03, 1.287416e-02, 5.641530e-03, 8.957260e-03, 0.000000e+00,
     -8.957260e-03, -5.641530e-03, -1.287416e-02,
     6.125880e-03]]], dtype=np.float32)).reshape((2, 1, 9, 9))
}
