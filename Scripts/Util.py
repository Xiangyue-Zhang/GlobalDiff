
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


import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
# import kornia.geometry as G
from typing import Union
import math
import torch.nn as nn


def SplitIntoOverlapedWindows(seq, win_size, stride_step, keep_last=False):
    assert stride_step <= win_size
    overlap_size = win_size - stride_step
    num = (len(seq) - win_size) // stride_step + 1
    res = [seq[i * stride_step:i * stride_step + win_size] for i in range(num)]
    if keep_last:
        last_id = num * stride_step
        if last_id < len(seq):
            res.append(seq[last_id:])
    return res


def matrix_to_rotation_6d(matrix):
    return matrix[..., :2, :].reshape((matrix.shape[:-2] + (6,)))


def Normalize(x, eps=1e-12):
    len = np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
    return x / (len + eps)


# def NormalizeToRot6d(d):
#     origin_shape = d.shape
#     d6 = d.reshape((-1, 6))
#     a1, a2 = d6[..., :3], d6[..., 3:]
#     b1 = F.normalize(a1, dim=-1)
#     b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
#     b2 = F.normalize(b2, dim=-1)
#     d6 = torch.cat([b1, b2], dim=-1)
#     return d6.reshape(origin_shape)


def rotation_6d_to_matrix_torch(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def rotation_6d_to_matrix_numpy(d6: torch.Tensor):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = Normalize(a1)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = Normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def rotation_6d_to_matrix(d6: Union[torch.Tensor, np.ndarray]):
    if isinstance(d6, torch.Tensor):
        return rotation_6d_to_matrix_torch(d6=d6)
    elif isinstance(d6, np.ndarray):
        return rotation_6d_to_matrix_numpy(d6=d6)
    else:
        assert 0 == 1


def rotation_matrix_to_axis_angle_torch(rot_mat: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    将旋转矩阵转换为轴角表示
    Args:
        rot_mat: (..., 3, 3) 旋转矩阵（必须有效正交矩阵）
        eps: 数值稳定性阈值
    Returns:
        axis_angle: (..., 3) 轴角表示（轴方向为单位向量，角度为向量的模长）
    """
    # 确保输入是3x3矩阵
    assert rot_mat.shape[-2:] == (3, 3), f"Invalid rotation matrix shape: {rot_mat.shape}"

    # 计算旋转角度θ（通过矩阵的迹）
    trace = rot_mat[..., 0, 0] + rot_mat[..., 1, 1] + rot_mat[..., 2, 2]
    theta = torch.acos(torch.clamp((trace - 1) / 2, -1 + eps, 1 - eps))
    # 计算轴方向
    axis = torch.empty_like(rot_mat[..., 0])
    # Case 1: θ ≈ 0（返回零向量）
    zero_angle = theta < eps
    axis[zero_angle] = torch.tensor([1.0, 0.0, 0.0], device=rot_mat.device)  # 默认轴

    # Case 2: θ ≈ π（需要特殊处理奇异性）
    near_pi = theta > math.acos(-1 + 2*eps)
    if torch.any(near_pi):
        # 从对角线元素计算轴
        r_diag = rot_mat[near_pi].diagonal(dim1=-2, dim2=-1)
        axis_near_pi = torch.sqrt( torch.clip((r_diag + 1) / 2, min=0.0) )
        # 符号需要根据非对角元素确定
        mask_x = ((rot_mat[near_pi, 2, 1] - rot_mat[near_pi, 1, 2]) > 0) * 2 - 1
        mask_y = ((rot_mat[near_pi, 0, 2] - rot_mat[near_pi, 2, 0]) > 0) * 2 - 1
        mask_z = ((rot_mat[near_pi, 1, 0] - rot_mat[near_pi, 0, 1]) > 0) * 2 - 1
        mask = torch.stack([mask_x, mask_y, mask_z], dim=-1)
        axis[near_pi] = axis_near_pi * mask


    # Case 3: 常规角度
    normal_angle = ~(zero_angle | near_pi)
    if torch.any(normal_angle):
        rot_mat_normal = rot_mat[normal_angle]
        axis[normal_angle] = torch.stack([
            rot_mat_normal[..., 2, 1] - rot_mat_normal[..., 1, 2],
            rot_mat_normal[..., 0, 2] - rot_mat_normal[..., 2, 0],
            rot_mat_normal[..., 1, 0] - rot_mat_normal[..., 0, 1]
        ], dim=-1) / (2 * torch.sin(theta[normal_angle].unsqueeze(-1)))

    # 将角度编码为向量的模长
    return axis * theta.unsqueeze(-1)



def rotation_matrix_to_axis_angle_np(matrix):
    """
    將旋轉矩陣轉換為軸角表示
    輸入: (*, 3, 3) 的旋轉矩陣 (numpy.ndarray)
    輸出: (*, 3) 的軸角表示，其中向量的方向是旋轉軸，範數是旋轉角度（弧度）
    """
    # 確保輸入是 numpy.ndarray
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix, dtype=np.float32)
    
    original_shape = matrix.shape[:-2]
    matrix = matrix.reshape(-1, 3, 3)  # 展平批次維度
    
    # 計算旋轉角度
    trace = np.trace(matrix, axis1=1, axis2=2)
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    
    # 計算旋轉軸
    axis = np.stack([
        matrix[:, 2, 1] - matrix[:, 1, 2],
        matrix[:, 0, 2] - matrix[:, 2, 0],
        matrix[:, 1, 0] - matrix[:, 0, 1]
    ], axis=1)
    
    axis_norm = np.linalg.norm(axis, axis=1, keepdims=True)
    # 處理角度接近0的情況（避免除以0）
    mask = (angle > 1e-6).reshape(-1, 1)
    axis = axis / (axis_norm + 1e-10) * mask + (1 - mask) * np.array([1., 0., 0.])
    
    # 組合軸和角度
    axis_angle = axis * angle.reshape(-1, 1)
    
    # 恢復原始批次形狀
    return axis_angle.reshape(*original_shape, 3)

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

def axis_angle_to_rotation_matrix_torch(omega_vec, eps: float = 1e-6):
    """
    将李代数向量转换为旋转矩阵。
    Args:
        omega_vec: (..., 3), 李代数向量
    Returns:
        R: (..., 3, 3), 旋转矩阵
    """
    theta = torch.norm(omega_vec, dim=-1, keepdim=True)
    near_zero = theta.squeeze(-1) < eps
    k = hat(omega_vec)
    K = hat( omega_vec /  (theta + 1e-12))   
    I = torch.eye(3, device=omega_vec.device, dtype=omega_vec.dtype).repeat(*omega_vec.shape[:-1], 1, 1)
    theta_expanded = theta.unsqueeze(-1)
    sin_theta = torch.sin(theta_expanded)
    cos_theta = torch.cos(theta_expanded)

    # Rodrigues公式
    R = I + sin_theta * K + (1 - cos_theta) * (K @ K)

    # 小角度近似
    R_approx = I + k + 0.5 * (k @ k)
    R = torch.where(near_zero[..., None, None], R_approx, R)
    return R
# def axis_angle_to_rotation_matrix_torch(axis_angle):
#     """
#     將軸角表示轉換為旋轉矩陣
#     輸入: (*, 3) 的軸角表示，其中向量的方向是旋轉軸，範數是旋轉角度（弧度）
#     輸出: (*, 3, 3) 的旋轉矩陣
#     """
#     # 確保輸入是torch.Tensor
#     if not isinstance(axis_angle, torch.Tensor):
#         axis_angle = torch.tensor(axis_angle, dtype=torch.float32)
    
#     batch_shape = axis_angle.shape[:-1]
#     axis_angle = axis_angle.view(-1, 3)  # 展平批次維度
    
#     angle = torch.norm(axis_angle, dim=1, keepdim=True)
#     # 處理角度接近0的情況
#     mask = (angle.squeeze() > 1e-6).float()
#     axis = axis_angle / (angle + 1e-10)
    
#     # 獲取軸的分量
#     x = axis[:, 0]
#     y = axis[:, 1]
#     z = axis[:, 2]
    
#     # 計算叉積矩陣
#     zero = torch.zeros_like(x)
#     cross_prod_mat = torch.stack([
#         zero, -z, y,
#         z, zero, -x,
#         -y, x, zero
#     ], dim=1).view(-1, 3, 3)
    
#     # 計算外積
#     outer_prod_mat = torch.bmm(axis.unsqueeze(2), axis.unsqueeze(1))
    
#     # 計算旋轉矩陣: R = cosθI + (1-cosθ)aa^T + sinθ[a]×
#     I = torch.eye(3, device=axis_angle.device).unsqueeze(0).expand(axis_angle.size(0), -1, -1)
#     cos_theta = torch.cos(angle).unsqueeze(2)
#     sin_theta = torch.sin(angle).unsqueeze(2)
    
#     rotation_matrix = I * cos_theta + (1 - cos_theta) * outer_prod_mat + sin_theta * cross_prod_mat
    
#     # 處理角度為0的情況
#     rotation_matrix = rotation_matrix * mask.view(-1, 1, 1) + I * (1 - mask).view(-1, 1, 1)
    
#     # 恢復原始批次形狀
#     return rotation_matrix.view(*batch_shape, 3, 3)

def axis_angle_to_rotation_matrix_np(axis_angle):
    aa = Rotation.from_rotvec(axis_angle, degrees=False)
    aa = aa.as_matrix()
    return aa

def rotation_matrix_to_axis_angle(m: Union[np.ndarray, torch.Tensor], eps: float = 1e-4):
    ori_shape = m.shape
    m  = m.reshape((-1, 3, 3))
    if isinstance(m, torch.Tensor):
        res =  rotation_matrix_to_axis_angle_torch(m, eps=eps)
    elif isinstance(m, np.ndarray):
        res =  Rotation.from_matrix(m).as_rotvec(degrees=False)
    else:
        assert 0 == 1
    res = res.reshape(ori_shape[:-1])
    return res

def axis_angle_to_rotation_matrix(axis_angle: Union[np.ndarray, torch.Tensor], eps: float = 1e-4):
    assert axis_angle.shape[-1] == 3
    ori_shape = axis_angle.shape
    axis_angle = axis_angle.reshape((-1, 3))
    if isinstance(axis_angle, torch.Tensor):
        res =  axis_angle_to_rotation_matrix_torch(axis_angle, eps=eps)
    elif isinstance(axis_angle, np.ndarray):
        res =  axis_angle_to_rotation_matrix_np(axis_angle)
    else:
        assert 0 == 1

    res = res.reshape(ori_shape[:-1] + (3, 3))
    return res


def rotation_6d_to_axis_angle(m: Union[np.ndarray, torch.Tensor], eps: float = 1e-4):
    orig_shape = m.shape
    m = rotation_6d_to_matrix(m.reshape((-1, 6)))
    return rotation_matrix_to_axis_angle(m, eps=eps).reshape(orig_shape[:-1] + (orig_shape[-1] // 6 * 3,))


def  axis_angle_to_rotation_6d(m: Union[np.ndarray, torch.Tensor]):
    orig_shape = m.shape
    m = axis_angle_to_rotation_matrix(m.reshape((-1, 3)))
    return matrix_to_rotation_6d(m).reshape(orig_shape[:-1] + (orig_shape[-1] // 3 * 6,))

def NormalizeRot6d(m):
    orig_shape = m.shape
    d6 = m.reshape((-1, 6))
    a1, a2 = d6[..., :3], d6[..., 3:]

    if isinstance(m, np.ndarray):
        b1 = Normalize(a1)
        b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
        b2 = Normalize(b2)
        m = np.concatenate([b1, b2], axis=-1)
    elif isinstance(m, torch.Tensor):
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        m = torch.cat([b1, b2], dim=-1)

    m = m.reshape(orig_shape)
    return m





def rotation_matrix_to_quaternion_torch(matrix:torch.Tensor) -> torch.Tensor:
    """
    将旋转矩阵转换为四元数（支持batch输入）
    
    参数:
        matrix: (..., 3, 3) 旋转矩阵
        
    返回:
        quaternion: (..., 4) 四元数 [w, x, y, z]
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"输入矩阵的最后两个维度必须是3x3，但得到的是{tuple(matrix.shape[-2:])}")
    
    batch_dim = matrix.shape[:-2]
    m = matrix.view(-1, 3, 3)
    
    num_rot = m.shape[0]
    quat = torch.zeros((num_rot, 4), dtype=matrix.dtype, device=matrix.device)
    
    # 计算矩阵对角线元素的和
    diag = torch.diagonal(m, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1)
    
    # 根据迹的大小选择不同的计算方式
    case_mask = torch.zeros_like(trace, dtype=torch.long)
    case_mask = torch.where(trace > 0, 
                           torch.tensor(0, device=matrix.device), 
                           case_mask)
    case_mask = torch.where((diag[:, 0] > diag[:, 1]) & 
                           (diag[:, 0] > diag[:, 2]) & 
                           (trace <= 0), 
                           torch.tensor(1, device=matrix.device), 
                           case_mask)
    case_mask = torch.where((diag[:, 1] > diag[:, 2]) & 
                           (trace <= 0), 
                           torch.tensor(2, device=matrix.device), 
                           case_mask)
    case_mask = torch.where(trace <= 0, 
                           torch.tensor(3, device=matrix.device), 
                           case_mask)
    
    # 根据不同情况计算四元数
    case0 = (case_mask == 0)
    if case0.any():
        s = torch.sqrt(trace[case0] + 1.0) * 2  # s = 4 * qw
        quat[case0, 0] = 0.25 * s
        quat[case0, 1] = (m[case0, 2, 1] - m[case0, 1, 2]) / s
        quat[case0, 2] = (m[case0, 0, 2] - m[case0, 2, 0]) / s
        quat[case0, 3] = (m[case0, 1, 0] - m[case0, 0, 1]) / s
    
    case1 = (case_mask == 1)
    if case1.any():
        s = torch.sqrt(1.0 + m[case1, 0, 0] - m[case1, 1, 1] - m[case1, 2, 2]) * 2
        quat[case1, 0] = (m[case1, 2, 1] - m[case1, 1, 2]) / s
        quat[case1, 1] = 0.25 * s
        quat[case1, 2] = (m[case1, 0, 1] + m[case1, 1, 0]) / s
        quat[case1, 3] = (m[case1, 0, 2] + m[case1, 2, 0]) / s
    
    case2 = (case_mask == 2)
    if case2.any():
        s = torch.sqrt(1.0 + m[case2, 1, 1] - m[case2, 0, 0] - m[case2, 2, 2]) * 2
        quat[case2, 0] = (m[case2, 0, 2] - m[case2, 2, 0]) / s
        quat[case2, 1] = (m[case2, 0, 1] + m[case2, 1, 0]) / s
        quat[case2, 2] = 0.25 * s
        quat[case2, 3] = (m[case2, 1, 2] + m[case2, 2, 1]) / s
    
    case3 = (case_mask == 3)
    if case3.any():
        s = torch.sqrt(1.0 + m[case3, 2, 2] - m[case3, 0, 0] - m[case3, 1, 1]) * 2
        quat[case3, 0] = (m[case3, 1, 0] - m[case3, 0, 1]) / s
        quat[case3, 1] = (m[case3, 0, 2] + m[case3, 2, 0]) / s
        quat[case3, 2] = (m[case3, 1, 2] + m[case3, 2, 1]) / s
        quat[case3, 3] = 0.25 * s
    
    # 归一化四元数
    quat = nn.functional.normalize(quat, p=2, dim=-1)
    
    # 恢复原始batch维度
    quat = quat.view(*batch_dim, 4)
    
    return quat


def slerp_torch(q0:torch.Tensor, q1:torch.Tensor, t:torch.Tensor):
    """
    四元数球面线性插值（slerp），支持batch输入
    
    参数:
        q0: (..., 4) 起始四元数 [w, x, y, z]
        q1: (..., 4) 结束四元数 [w, x, y, z]
        t: (...) 插值系数，范围[0, 1]
        
    返回:
        q: (..., 4) 插值后的四元数 [w, x, y, z]
    """
    # 确保输入形状一致
    assert q0.shape == q1.shape, "q0和q1的形状必须相同"
    
    # 确保四元数是单位四元数
    q0 = nn.functional.normalize(q0, p=2, dim=-1)
    q1 = nn.functional.normalize(q1, p=2, dim=-1)
    
    # 计算点积（即cos(omega)）
    dot = (q0 * q1).sum(dim=-1, keepdim=True)  # (..., 1)
    
    # 如果点积为负，取反其中一个四元数以保证走最短路径
    neg_mask = dot < 0.0
    q1 = torch.where(neg_mask, -q1, q1)
    dot = torch.where(neg_mask, -dot, dot)
    
    # 限制dot在[-1,1]范围内，防止数值误差
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # 计算角度和sin(omega)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    
    # 处理sin(omega)接近0的情况
    eps = 1e-8
    small_mask = sin_omega < eps
    sin_omega = torch.where(small_mask, torch.ones_like(sin_omega), sin_omega)
    
    # 计算插值权重
    t = t.unsqueeze(-1)  # (..., 1)
    w0 = torch.where(small_mask, 1.0 - t, torch.sin((1.0 - t) * omega) / sin_omega)
    w1 = torch.where(small_mask, t, torch.sin(t * omega) / sin_omega)
    
    # 线性组合
    q = w0 * q0 + w1 * q1
    
    # 归一化结果
    q = nn.functional.normalize(q, p=2, dim=-1)
    
    return q



def quaternion_to_matrix_torch(quaternions):
    """
    将四元数转换为旋转矩阵（支持batch输入）
    
    参数:
        quaternions: (..., 4) 四元数，格式为[w, x, y, z]，其中w是实部
        
    返回:
        rotation_matrix: (..., 3, 3) 对应的旋转矩阵
    """
    if quaternions.size(-1) != 4:
        raise ValueError(f"输入四元数的最后一个维度必须是4，但得到的是{quaternions.shape[-1]}")
    
    # 归一化四元数
    quaternions = nn.functional.normalize(quaternions, p=2, dim=-1)
    
    # 提取四元数分量
    w, x, y, z = torch.unbind(quaternions, dim=-1)
    
    # 计算平方项
    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w
    
    # 计算交叉项
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    
    # 构建旋转矩阵
    # 第一列
    m00 = w2 + x2 - y2 - z2
    m10 = 2 * (xy + wz)
    m20 = 2 * (xz - wy)
    
    # 第二列
    m01 = 2 * (xy - wz)
    m11 = w2 - x2 + y2 - z2
    m21 = 2 * (yz + wx)
    
    # 第三列
    m02 = 2 * (xz + wy)
    m12 = 2 * (yz - wx)
    m22 = w2 - x2 - y2 + z2
    
    # 组合成旋转矩阵
    rotation_matrix = torch.stack([
        torch.stack([m00, m01, m02], dim=-1),
        torch.stack([m10, m11, m12], dim=-1),
        torch.stack([m20, m21, m22], dim=-1)
    ], dim=-2)
    
    return rotation_matrix

if __name__ == '__main__':
    import time
    x = Rotation.from_euler('xyz', [0, 0, 180-0.1], degrees=True).as_matrix()[None].astype(np.float32)
    x_tensor = torch.from_numpy(x)

    aa_tensor = rotation_matrix_to_axis_angle(x_tensor)
    print(aa_tensor)

    aa = rotation_matrix_to_axis_angle(x)
    print(aa)


    r_tensor = axis_angle_to_rotation_matrix(aa_tensor)
    r = axis_angle_to_rotation_matrix(aa)

    d = r_tensor@torch.from_numpy(r).float()

    d = rotation_matrix_to_axis_angle(d)
    print(torch.linalg.norm(d, dim=-1) * 180 / math.pi)