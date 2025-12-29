
'''
Copyright (c) 2021, Alibaba Cloud and its affiliates;
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import  torch
import  random
import  numpy as np
import  math
import sys
sys.path.append('..')
from Util import  rotation_matrix_to_axis_angle, axis_angle_to_rotation_matrix




def hat(v: torch.Tensor) -> torch.Tensor:
    """
    将向量转换为反对称矩阵。
    Args:
        v: (..., 3), 李代数向量
    Returns:
        K: (..., 3, 3), 反对称矩阵
    """
    K = torch.zeros((*v.shape[:-1], 3, 3), device=v.device, dtype=v.dtype)
    K[..., 0, 1] = -v[..., 2]
    K[..., 0, 2] = v[..., 1]
    K[..., 1, 0] = v[..., 2]
    K[..., 1, 2] = -v[..., 0]
    K[..., 2, 0] = -v[..., 1]
    K[..., 2, 1] = v[..., 0]
    return K



def matrix_log(delta: torch.Tensor) -> torch.Tensor:
    return rotation_matrix_to_axis_angle(delta)


def matrix_exp(omega_vec: torch.Tensor) -> torch.Tensor:
    return axis_angle_to_rotation_matrix(omega_vec)


def slerp(R0: torch.Tensor, R1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    球面线性插值。
    Args:
        R0: (..., 3, 3), 起始旋转矩阵
        R1: (..., 3, 3), 终止旋转矩阵
        t: 插值系数，范围[0, 1]
    Returns:
        R_t: (..., 3, 3), 插值后的旋转矩阵
    """
    # while len(t.shape) < len(R0.shape):
    assert len(t.shape)  + 2 == len(R0.shape)

    delta = torch.matmul(R0.transpose(-2, -1), R1)
    omega_vec = matrix_log(delta)
    t_omega_vec = t.unsqueeze(-1) * omega_vec
    R_t = torch.matmul(R0, matrix_exp(t_omega_vec))
    return R_t


def dRdt(R0: torch.Tensor, R1:torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    计算插值旋转矩阵在时间t处的导数。
    Args:
        t: 插值系数
        R0: (..., 3, 3), 起始旋转矩阵
        R1: (..., 3, 3), 终止旋转矩阵
    Returns:
        dR_dt: (..., 3, 3), 导数矩阵
    """

    assert len(t.shape)  + 2 == len(R0.shape)
    delta = torch.matmul(R0.transpose(-2, -1), R1)
    omega_vec = matrix_log(delta)
    Omega = hat(omega_vec)
    t_omega_vec = t.unsqueeze(-1) * omega_vec
    exp_t_omega = matrix_exp(t_omega_vec)
    dR_dt = torch.matmul(R0, torch.matmul(Omega, exp_t_omega))
    return dR_dt



def euler(R0, omega_vec, h):
    return R0 @ matrix_exp(omega_vec * h)






if __name__ == '__main__':
    from scipy.spatial.transform import Rotation
    import math
    np.random.seed(0)

    # r = np.array([[ 9.3594e-01, -3.5215e-01,  9.3822e-05],
    #     [-3.5215e-01, -9.3594e-01,  2.6652e-04],
    #     [-6.2024e-06, -2.8247e-04, -1.0000e+00]])
    r = np.load('nan3.npy')
    r = torch.from_numpy(r).float()[None]
    omega_vec = matrix_log(r)
    print(omega_vec)
    