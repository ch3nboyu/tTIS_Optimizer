# -*- coding: utf-8 -*-
"""
Toolkit for optimizing or simulating transcranial temporal Inference Stimulation (tTIS)

origin: https://github.com/simnibs/simnibs/blob/master/simnibs/utils/TI_utils.py

@modifier: boyu
@Date: 2024-12-27 14:57:19
"""

import h5py
import numpy as np
import cupy as cp

from simnibs.mesh_tools import mesh_io


def load_leadfield(leadfield_hdf,
                   leadfield_path = '/mesh_leadfield/leadfields/tdcs_leadfield',
                   mesh_path = '/mesh_leadfield/'):
    """a
    load leadfield, mesh on which leadfield was calculated and mapping from 
    electrode names to index in the leadfield

    Parameters
    ----------
    leadfield_hdf : string
        path to the leadfield hdf5 file
    leadfield_path : string, optional
        path inside the hdf5 file to the leadfield. 
        The default is '/mesh_leadfield/leadfields/tdcs_leadfield'.
    mesh_path : string, optional
        path inside the hdf5 file to the mesh.
        The default is '/mesh_leadfield/'.

    Returns
    -------
    leadfield : np.ndarray
        Leadfield matrix (N_elec -1 x M x 3) where M is either 
        the number of nodes (for surface-based leadfields) or 
        the number of elements (tet-based leadfields) in the mesh.
    mesh : simnibs.msh.mesh_io.Msh
        Mesh on which the leadfield was calculated
    idx_lf : dict
        Mapping from electrode name to index in the leadfield matrix. The 
        reference electrode has None as index
    """
    with h5py.File(leadfield_hdf, 'r') as f:
        lf_struct = f[leadfield_path]
        leadfield = lf_struct[:] # elecs x mesh nodes x 3
        
        # make a dict: elec name --> index in leadfield matrix
        name_elecs = lf_struct.attrs.get('electrode_names')
        name_ref = lf_struct.attrs.get('reference_electrode')
        name_elecs = name_elecs[name_elecs != name_ref]
        assert leadfield.shape[0] == len(name_elecs) 
        idx_lf = dict(zip(name_elecs, range(len(name_elecs))))
        idx_lf[name_ref] = None
        
    mesh = mesh_io.Msh.read_hdf5(leadfield_hdf, mesh_path) # usually the two hemispheres and eyes
    
    return leadfield, mesh, idx_lf
 

def get_field(elec_pair, leadfield, idx_lf):
    """
    return field for a specific electrode pair and current intensity

    Parameters
    ----------
    elec_pair : list [elec_1,elec_2,current_intensity]
        specifies an electrode pair and the current intensity into first electrode
        (current in second electrode: -current_intensity)
    leadfield : np.ndarray
        Leadfield matrix (N_elec -1 x M x 3) where M is either 
        the number of nodes (for surface-based leadfields) or 
        the number of elements (tet-based leadfields) in the mesh.
    idx_lf : dict
        Mapping from electrode name to index in the leadfield matrix. The 
        reference electrode has None as index

    Returns
    -------
    np.ndarray
       field matrix (M x 3) where M is either the number of 
       nodes (for surface-based leadfields) or the number of elements 
       (tet-based leadfields) in the mesh.

    """
    assert elec_pair[0] != elec_pair[1]
    if idx_lf[elec_pair[0]] is None:
        return -elec_pair[2]*leadfield[ idx_lf[elec_pair[1]] ]
    if idx_lf[elec_pair[1]] is None:
        return  elec_pair[2]*leadfield[ idx_lf[elec_pair[0]] ]
    return elec_pair[2]*(leadfield[idx_lf[elec_pair[0]]] - leadfield[idx_lf[elec_pair[1]] ])


def get_field_gpu(elec_pair, leadfield, idx_lf):
    """
    return field for a specific electrode pair and current intensity (GPU version using CuPy)

    Parameters: elec_pair : list [elec_1, elec_2, current_intensity]
                leadfield : np.ndarray
                idx_lf : dict

    Returns: cp.ndarray
    """
    assert elec_pair[0] != elec_pair[1]
    
    # Ensure leadfield is a CuPy array (if not, convert it)
    if not isinstance(leadfield, cp.ndarray):
        leadfield = cp.asarray(leadfield)
    
    if idx_lf[elec_pair[0]] is None:
        return -elec_pair[2] * leadfield[ idx_lf[elec_pair[1]] ]
    if idx_lf[elec_pair[1]] is None:
        return elec_pair[2] * leadfield[ idx_lf[elec_pair[0]] ]
    return elec_pair[2] * (leadfield[idx_lf[elec_pair[0]]] - leadfield[idx_lf[elec_pair[1]] ])


def get_maxTI(E1_org, E2_org):
    """
    calculates the maximal modulation amplitude of the TI envelope using 
    the equation given in Grossman et al, Cell 169, 1029–1041.e6, 2017

    Parameters
    ----------
    E1 : np.ndarray
           field of electrode pair 1 (N x 3) where N is the number of 
           positions at which the field was calculated
    E2 : np.ndarray
        field of electrode pair 2 (N x 3)

    Returns
    -------
    TImax : np.ndarray (N,)
        maximal modulation amplitude

    """
    # 检查输入数组的形状是否一致，并且第二维度为 3（表示三维向量）
    assert E1_org.shape == E2_org.shape
    assert E1_org.shape[1] == 3

    # 复制输入数组，避免修改原始数据
    E1 = E1_org.copy()
    E2 = E2_org.copy()
    
    # 确保 E1 的范数大于 E2 的范数
    idx = np.linalg.norm(E2, axis=1) > np.linalg.norm(E1, axis=1)
    E1[idx] = E2[idx]  # 如果 E2 的范数更大，交换 E1 和 E2
    E2[idx] = E1_org[idx]

    # 确保两个电场向量的夹角 alpha < pi/2
    idx = np.sum(E1 * E2, axis=1) < 0  # 如果点积为负，说明夹角大于 pi/2
    E2[idx] = -E2[idx]  # 将 E2 取反，使夹角小于 pi/2
    
    # 计算包络的最大幅度
    normE1 = np.linalg.norm(E1, axis=1)  # E1 的范数
    normE2 = np.linalg.norm(E2, axis=1)  # E2 的范数
    cosalpha = np.sum(E1 * E2, axis=1) / (normE1 * normE2)  # 计算夹角的余弦值
    
    # 根据公式计算最大调制幅度
    TImax = 2 * np.linalg.norm(np.cross(E2, E1 - E2), axis=1) \
            / np.linalg.norm(E1 - E2, axis=1)
    
    # 如果 E2 的范数小于等于 E1 的范数乘以 cosalpha，调整 TImax
    idx = normE2 <= normE1 * cosalpha
    TImax[idx] = 2 * normE2[idx]
    
    return TImax


def get_maxTI_gpu(E1_org, E2_org):
    """
    calculates the maximal modulation amplitude of the TI envelope using 
    the equation given in Grossman et al, Cell 169, 1029–1041.e6, 2017 (GPU version using CuPy)

    Parameters: E1_org : cp.ndarray
                E2_org : cp.ndarray

    Returns: TImax : cp.ndarray (N,)
    """
    assert E1_org.shape == E2_org.shape
    assert E1_org.shape[1] == 3
    
    # Ensure is CuPy array (if not, convert it)
    if not isinstance(E1_org, cp.ndarray):
        E1_org = cp.asarray(E1_org)
    if not isinstance(E2_org, cp.ndarray):
        E2_org = cp.asarray(E2_org)

    E1 = E1_org.copy()
    E2 = E2_org.copy()
    
    # ensure E1 > E2
    idx = cp.linalg.norm(E2, axis=1) > cp.linalg.norm(E1, axis=1)
    E1[idx] = E2[idx]
    E2[idx] = E1_org[idx]

    # ensure alpha < pi/2
    idx = cp.sum(E1 * E2, axis=1) < 0
    E2[idx] = -E2[idx]
    
    # get maximal amplitude of envelope
    normE1 = cp.linalg.norm(E1, axis=1)
    normE2 = cp.linalg.norm(E2, axis=1)
    cosalpha = cp.sum(E1 * E2, axis=1) / (normE1 * normE2)
    
    TImax = 2 * cp.linalg.norm(cp.cross(E2, E1 - E2), axis=1) \
            / cp.linalg.norm(E1 - E2, axis=1)
    idx = normE2 <= normE1 * cosalpha
    TImax[idx] = 2 * normE2[idx]
    return TImax


def get_dirTI(E1, E2, dirvec_org):
    """
    计算沿向量n指定的方向上的TI包络振幅
    
    TIamp = | |(E1+E2)*n| - |(E1-E1)*n| |
    
    参数
    ----------
    E1 : np.ndarray
        电极对1的场（N x 3），其中N是计算场的位置数
    E2 : np.ndarray
        电极对2的场（N x 3）
    dirvec_org : np.ndarray 或 list
        可以是单个向量（1 x 3），应用于所有位置，
        也可以是每个位置一个向量（N x 3）
    
    返回值
    -------
    TIamp : np.ndarray
        沿n指定的方向上的调制振幅
    """
    
    assert E1.shape == E2.shape
    assert E1.shape[1] == 3
    
    dirvec = np.array(dirvec_org)
    if dirvec.ndim == 1:
        assert len(dirvec) == 3
        dirvec = dirvec.reshape((1,3))
    if dirvec.shape[0] > 1:
        assert dirvec.shape == E1.shape
    dirvec = dirvec/np.linalg.norm(dirvec,axis=1)[:,None]

    TIamp = np.abs( np.abs(np.sum((E1+E2)*dirvec,axis=1)) - np.abs(np.sum((E1-E2)*dirvec,axis=1)) )
    return TIamp


def get_dirTI_gpu(E1, E2, dirvec_org):
    """
    计算沿向量n指定的方向上的TI包络振幅 (GPU version using CuPy)

    Parameters
    ----------
    E1 : cp.ndarray
    E2 : cp.ndarray
    dirvec_org : cp.ndarray 或 list
        方向向量，可以是单个向量 (3,) 或多个向量 (N x 3)。

    Returns
    -------
    TIamp : cp.ndarray (N,)
    """
    # 检查输入数组的形状是否一致，并且第二维度为 3（表示三维向量）
    assert E1.shape == E2.shape
    assert E1.shape[1] == 3

    # 将方向向量转换为 CuPy 数组
    dirvec = cp.array(dirvec_org)

    # 如果方向向量是单个向量，将其形状调整为 (1, 3)
    if dirvec.ndim == 1:
        assert len(dirvec) == 3
        dirvec = dirvec.reshape((1, 3))

    # 如果方向向量的数量与电场位置数量不一致，抛出异常
    if dirvec.shape[0] > 1:
        assert dirvec.shape == E1.shape

    # 归一化方向向量
    dirvec = dirvec / cp.linalg.norm(dirvec, axis=1)[:, None]

    # 计算方向性 TI 幅度
    TIamp = cp.abs(
        cp.abs(cp.sum((E1 + E2) * dirvec, axis=1)) -
        cp.abs(cp.sum((E1 - E2) * dirvec, axis=1))
    )

    return TIamp
