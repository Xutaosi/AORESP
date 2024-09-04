import cupy as cp
import numpy as np
from scipy.spatial.transform import Rotation
import scipy.linalg
import time

def gen_data(total_num, outlier_num, noise_level):
    R_gt = Rotation.random(1)
    inlier_num = total_num - outlier_num
    x_data_non_normal = cp.random.rand(total_num, 3) * 2 - 1
    x_data = x_data_non_normal / cp.linalg.norm(x_data_non_normal, axis=1, keepdims=True)
    y_data_inlier_non_normal = cp.asarray(R_gt.apply(cp.asnumpy(x_data[0:inlier_num, :]))) + noise_level * cp.random.randn(inlier_num, 3)
    y_data_outlier_non_normal = cp.random.rand(outlier_num, 3) * 2 - 1
    y_data_non_normal = cp.vstack((y_data_inlier_non_normal, y_data_outlier_non_normal))
    y_data = y_data_non_normal / cp.linalg.norm(y_data_non_normal, axis=1, keepdims=True)
    axis_ = R_gt.as_rotvec()
    angle_gt = scipy.linalg.norm(axis_)
    axis_gt = axis_ / angle_gt
    return cp.asnumpy(x_data), cp.asnumpy(y_data), R_gt, axis_gt, angle_gt

def calculate_voting(x_input, y_input):
    z_non_normal = x_input - y_input
    z_unit = z_non_normal / cp.linalg.norm(z_non_normal, axis=1, keepdims=True)
    any_vector_non_normal = cp.random.rand(1, 3) * 2 - 1
    any_unit = any_vector_non_normal / cp.linalg.norm(any_vector_non_normal)
    a1_non_unit = any_unit - cp.tensordot(z_unit, any_unit, axes=([1], [1])) * z_unit
    a1 = a1_non_unit / cp.linalg.norm(a1_non_unit, axis=1, keepdims=True)
    a2 = cp.cross(a1, z_unit, axis=1)
    theta_num = 180
    theta = cp.linspace(0, cp.pi, theta_num, endpoint=True)
    cos_theta = cp.cos(theta)
    sin_theta = cp.sin(theta)
    bins_num = 180+1 
    bins = cp.linspace(-1, 1, bins_num, endpoint=True)
    batch_num = 5000
    divided_num = total_num // batch_num
    part_list = cp.array_split(cp.arange(a1.shape[0]), divided_num)
    accumulated=cp.zeros((bins_num-1,bins_num-1))
    for index in part_list:
        accumulators = cp.zeros((len(index), bins_num - 1, bins_num - 1), dtype=cp.int32)
        A1 = cp.broadcast_to(a1[index, :], (theta_num, len(index), 3))
        A2 = cp.broadcast_to(a2[index, :], (theta_num, len(index), 3))
        cos_theta_list = cos_theta.reshape(theta_num, 1, 1)
        sin_theta_list = sin_theta.reshape(theta_num, 1, 1)
        points_3d = A1 * cos_theta_list + A2 * sin_theta_list
        ind_neg = cp.where(points_3d[:, :, 2] < 0)
        points_3d[ind_neg[0], ind_neg[1], :] *= -1
        point_2d_all=points_3d[:, :, 0:2] / (1 + points_3d[:, :, 2:3])
        each_bin_length=bins[1]-bins[0]
        ind_xyz=cp.floor((point_2d_all+1)/each_bin_length).astype(cp.int64)
        len_index=len(index)
        ind_x=cp.broadcast_to(cp.arange(len_index),(theta_num,len_index)).ravel(order='F')
        ind_y=ind_xyz[:,cp.arange(len_index),0].ravel(order='F')
        ind_z=ind_xyz[:,cp.arange(len_index),1].ravel(order='F')
        accumulators[ind_x,ind_y,ind_z]=1
        accumulated += cp.sum(accumulators, axis=0)
    ind = cp.unravel_index(cp.argmax(accumulated), accumulated.shape)
    axis_2d_1 = 0.5 * (bins[ind[0]] + bins[ind[0] + 1])
    axis_2d_2 = 0.5 * (bins[ind[1]] + bins[ind[1] + 1])
    fd = 1 + axis_2d_1 * axis_2d_1 + axis_2d_2 * axis_2d_2
    est_axis = cp.array([2 * axis_2d_1 / fd, 2 * axis_2d_2 / fd, 2 / fd - 1])
    return est_axis

def estimate_angle(given_rot_axis, x_input, y_input):
    given_rot_axis = cp.asarray(given_rot_axis)
    rot_axis_all = cp.broadcast_to(given_rot_axis, (x_input.shape[0], 3))
    x_prime_non_unit = cp.cross(rot_axis_all, x_input, axis=1)
    y_prime_non_unit = cp.cross(rot_axis_all, y_input, axis=1)
    x_prime = x_prime_non_unit / cp.linalg.norm(x_prime_non_unit, axis=1, keepdims=True)
    y_prime = y_prime_non_unit / cp.linalg.norm(y_prime_non_unit, axis=1, keepdims=True)
    angle_all_candiate = cp.arccos(cp.sum(x_prime * y_prime, axis=1, keepdims=True))
    z_prime = cp.cross(x_prime, y_prime, axis=1)
    ind = cp.sum(z_prime * rot_axis_all, axis=1) < 0
    angle_all_candiate[ind] *= -1
    H, bin_edges = cp.histogram(angle_all_candiate, bins=cp.linspace(-cp.pi, cp.pi, num=180 + 1))
    ind = cp.argmax(H)
    est_angle = 0.5 * (bin_edges[ind] + bin_edges[ind + 1])
    return est_angle

def est_rotation(axis, angle, x_in, y_in):
    cos_theta = cp.cos(angle)
    sin_theta = cp.sin(angle)
    u_dot_x = cp.sum(axis * x_in, axis=1, keepdims=True)
    u_cross_x = cp.cross(axis, x_in)
    x_tran = cos_theta * x_in + (1 - cos_theta) * u_dot_x* axis + sin_theta * u_cross_x
    ind_inlier = cp.linalg.norm((x_tran - y_in), axis =1) < 0.1
    M = y_in[ind_inlier, :].T@ x_in[ind_inlier, :]
    u, s, v = cp.linalg.svd(M)
    if cp.linalg.det(u@v) > 0:
        S = cp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    else:
        S = cp.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    R_m = u@S@v
    return R_m

def est_rotation_op(R_est, x_input, y_input):
    global R_gt
    iterations = 2
    for _ in range(iterations):
        y_data_2 = x_input @ cp.asarray(R_est).T
        ind_inlier = cp.linalg.norm((y_input - y_data_2), axis =1) < 0.1
        M = y_input[ind_inlier, :].T@ x_input[ind_inlier, :]
        u, s, v = cp.linalg.svd(M)
        if cp.linalg.det(u@v) > 0:
            S = cp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        else:
            S = cp.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        R_n = u@S@v
        R_est = R_n
    R_est_end = R_est
    return R_est_end

def calc_err(R_m, R_gt):
    R_m = cp.asarray(R_m)
    R_gt = cp.asarray(R_gt)
    R_product_np = cp.asnumpy(R_m.T @ R_gt)
    ERR = Rotation.from_matrix(R_product_np)
    err = ERR.magnitude()[0] / np.pi * 180
    return err

def our_alg(x_data, y_data):
    x_data = cp.asarray(x_data)
    y_data = cp.asarray(y_data)
    est_axis = calculate_voting(x_data, y_data)
    est_angle = estimate_angle(est_axis, x_data, y_data)
    R_est = est_rotation(est_axis, est_angle, x_data, y_data)
    R_est_end = est_rotation_op(R_est, x_data, y_data)
    return R_est_end

if __name__ == '__main__':
    total_num = 100000
    noise_level = 0.01
    outlier_num = int(total_num * 0.5)
    inlier_num = total_num - outlier_num
    x_data, y_data, R_gt, axis_gt, angle_gt = gen_data(total_num, outlier_num, noise_level)
    x_data = cp.asarray(x_data)
    y_data = cp.asarray(y_data)
    start_tim = time.perf_counter_ns()
    R_est_end = our_alg(x_data, y_data)
    end_tim = time.perf_counter_ns()
    tim = (end_tim - start_tim) / 1000000000
    err = calc_err(R_est_end, R_gt.as_matrix())
    print("err:",err)
    print("tim:", tim)

