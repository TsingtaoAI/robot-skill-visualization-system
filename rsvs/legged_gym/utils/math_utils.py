import torch
from torch import Tensor
import numpy as np
from typing import Tuple

@torch.jit.script
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

@torch.jit.script
def quat_xyzw_to_wxyz(quat):
    return quat[:, [3, 0, 1, 2]]

@torch.jit.script
def quat_wxyz_to_xyzw(quat):
    return quat[:, [1, 2, 3, 0]]

@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat

@torch.jit.script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)

@ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

@ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

@ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower

@torch.jit.script
def quat_rotate_inverse(q, v):
    '''
    Rotate vector v by the inverse of quaternion q.
    q: shape (..., 4)
    v: shape (..., 3)'''
    shape = q.shape
    assert v.shape == shape[:-1] + (3,), f"Shape mismatch: q {shape}, v {v.shape}"
    q_w = q[..., 3]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w ** 2 - 1.0)[..., None]
    b = torch.cross(q_vec, v, dim=-1) * (2.0 * q_w)[..., None]
    c = q_vec * (2.0 * torch.sum(q_vec * v, dim=-1, keepdim=True))
    return a - b + c

def quat_rotate_inverse_np(q, v):
    '''
    Rotate vector v by the inverse of quaternion q.
    q: shape (..., 4)
    v: shape (..., 3)'''
    shape = q.shape
    assert v.shape == shape[:-1] + (3,), f"Shape mismatch: q {shape}, v {v.shape}"
    q_w = q[..., 3]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w ** 2 - 1.0)[..., None]
    b = np.cross(q_vec, v, axis=-1) * (2.0 * q_w)[..., None]
    c = q_vec * (2.0 * np.sum(q_vec * v, axis=-1, keepdims=True))
    return a - b + c

@torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return (upper - lower) * torch.rand(*shape, device=device) + lower

@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)

@torch.jit.script
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)

@torch.jit.script
def quat_slerp(q0, q1, t):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    # q: [..., 4] (x, y, z, w)
    # t: [0, 1]
    
    # standardize: ensure short path (dot > 0)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.abs(dot)
    
    # clamp for numerical stability
    dot = torch.clamp(dot, -1.0, 1.0)
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    
    # linear interpolation for very small angles to avoid division by zero
    mask = (sin_theta_0 < 1e-6).flatten()
    
    # lerp
    res_lerp = (1.0 - t) * q0 + t * q1
    
    # slerp
    theta = theta_0 * t
    sin_theta = torch.sin(theta)
    s0 = torch.cos(theta) - dot * sin_theta / (sin_theta_0 + 1e-9)
    s1 = sin_theta / (sin_theta_0 + 1e-9)
    res_slerp = s0 * q0 + s1 * q1
    
    res = torch.where(mask[..., None], res_lerp, res_slerp)
    return normalize(res)

@torch.jit.script
def standardize_quaternion(q):
    # type: (Tensor) -> Tensor
    # Ensure w is positive
    return torch.where(q[..., 3:4] < 0, -q, q)

@torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)

@torch.jit.script
def dr_normalize(x, min_val, max_val):
    # type: (Tensor, float, float) -> Tensor
    """ 根据输入的Tensor和最小值最大值范围, 将Tensor中的值归一化到-1到1之间。 
        normalization for domain randomization parameters, which may have different ranges.
    """
    range_val = max_val - min_val
    if abs(range_val) < 1e-12:
        return torch.zeros_like(x)

    normalized = 2 * (x - min_val) / range_val - 1
    return torch.clamp(normalized, -1.0, 1.0)