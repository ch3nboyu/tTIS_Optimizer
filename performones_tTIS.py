#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
this script performs one single tTIS using a given combination of electrodes
and return relevant information such as lf of ROI

@Author: boyu
@Date:   2024-10-21 17:20:25
"""
import os
import json
import time, copy
import numpy as np
import cupy as cp
import gmsh
import yaml
import simnibs
from simnibs import mesh_io
import utils.TI_utils as TI
from utils.utils import electrode_Pos
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Rectangle
import scipy.io as sio
from utils.utils import electrode_Pos, load_cfg
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import gmsh
from tqdm import tqdm

def read_and_save_msh(input_file_path, output_file_path):
    gmsh.initialize()
    
    if input_file_path.endswith('.geo'):
        gmsh.open(input_file_path)
        gmsh.model.geo.synchronize()
    elif input_file_path.endswith('.msh'):
        gmsh.merge(input_file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .geo or .msh file.")
    
    gmsh.write(output_file_path)
    
    gmsh.finalize()

def plotElec1010(U, grayMark):
    pos = electrode_Pos()
    elecXY, elecName = pos.id2pos, pos.id2elec  # 假设此函数已定义并返回电极坐标和名称

    # 找到每个电极在绘图中的索引
    elecFigureIdx = pos.elec2id
    elec_pairs = []
    for pair in U:
        elec_pairs.append([elecFigureIdx[e] for e in pair['elecName']])

    # 定义电极颜色
    minCurrent, maxCurrent, stepCurrent = -2, 2, 0.05
    current = np.arange(minCurrent, maxCurrent + stepCurrent, stepCurrent)
    Ncolor = len(current)

    if grayMark == 1:
        colorMap = plt.cm.Greys(np.linspace(0, 1, Ncolor))  # 使用灰色调颜色图
    else:
        colorMap = plt.cm.Spectral(np.linspace(0, 1, Ncolor))  # 使用分隔型颜色图（例如红蓝对比）

    # 线性插值以根据实际电流强度分配颜色
    elec_color = []
    for pair in U:
        # 确保颜色为RGBA格式
        colors = np.array([colorMap[int((cu - minCurrent) / stepCurrent)] for cu in pair['current']])
        if colors.shape[1] == 3:
            colors = np.hstack((colors, np.ones((colors.shape[0], 1))))  # 添加Alpha通道
        elec_color.append(colors)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.axis('off')

    # 绘制头部轮廓
    lineWidHead, lineWidTarget, elec_r = 1, 2, 0.33

    # 绘制头部圆圈和十字线
    circle_5 = Circle((0, 0), 5, edgecolor='k', facecolor='none', linewidth=lineWidHead)
    circle_4 = Circle((0, 0), 4, edgecolor='k', facecolor='none', linewidth=lineWidHead)
    ax.add_patch(circle_5)
    ax.add_patch(circle_4)
    ax.plot([-5, 5], [0, 0], 'k-', lw=lineWidHead, zorder=1)
    ax.plot([0, 0], [-5, 5], 'k-', lw=lineWidHead, zorder=1)
    ax.plot([-0.5, 0], [np.sqrt(25-0.5**2), 5.5], 'k-', lw=lineWidHead, zorder=1)
    ax.plot([0.5, 0], [np.sqrt(25-0.5**2), 5.5], 'k-', lw=lineWidHead, zorder=1)

    # 绘制电极
    elecIdxOther = set(range(1, len(elecName)+1)).difference(tuple(e) for e in elec_pairs)
    for i in elecIdxOther:
        ax.add_patch(Circle(elecXY[str(i)], elec_r, edgecolor='k', facecolor='w', linewidth=lineWidHead))
        ax.text(elecXY[str(i)][0], elecXY[str(i)][1], elecName[i], ha='center', va='center', fontsize=7)

    for i_pair, pair in enumerate(elec_pairs):
        for i, idx in enumerate(pair):
            ax.add_patch(Circle(elecXY[str(idx)], elec_r, edgecolor='k', facecolor=elec_color[i_pair][i], linewidth=lineWidTarget))
            ax.text(elecXY[str(idx)][0], elecXY[str(idx)][1], elecName[idx], ha='center', va='center', fontsize=7)

    # 添加颜色条
    norm = plt.Normalize(minCurrent, maxCurrent)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Spectral_r if not grayMark else plt.cm.Greys, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='Current Intensity')

    plt.show()


def coord_sub2clip_str(coord):
    """
    将坐标转换为切片平面字符串列表
    参数:
    coord (list or numpy.ndarray): 包含x, y, z坐标的列表或数组。
    """
    clip_str = []
    
    # 遍历坐标并构造切片字符串
    for i in range(3):
        clip_str.append(round(coord[i]))
    return clip_str


def normalize_to_range(data, new_min=-0.2, new_max=0.2):
    """Normalize data to a specified range [new_min, new_max]."""
    old_min, old_max = np.min(data), np.max(data)
    return new_min + (data - old_min) * (new_max - new_min) / (old_max - old_min)


def plot_elec_clip(cfg, electrodepair, TImax, mesh):
    U = []
    for i in range(len(electrodepair)):
        elecNum = 2
        elecName = electrodepair[i][:2]
        current = [electrodepair[i][2], -electrodepair[i][2]]
        U.append(
            {"elecNum": elecNum, "elecName": elecName, "current": current}
        )
    plotElec1010(U, 0)
    
    case = cfg['case']
    if case == 1:
        ROIcfg = cfg['case1']
    elif case == 2:
        ROIcfg = cfg['case2']
    ROIcoordMNI = ROIcfg['coordMNI']
    clipStr = coord_sub2clip_str(ROIcoordMNI[0])

    # plot_clip(cfg, U, 0.2, clipStr, 2, TImax, mesh)


def cut_model_at_point(file_path, position):
    gmsh.initialize()

    if file_path.endswith('.geo'):
        gmsh.open(file_path)
        gmsh.model.geo.synchronize()
    elif file_path.endswith('.msh'):
        gmsh.merge(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .geo or .msh file.")

    elementTypes, elementTags, _ = gmsh.model.mesh.getElements()
    nodeTags, coords, parametricCoords = gmsh.model.mesh.getNodes(includeBoundary=True)
    
    # print(coords)


    # for elementTag in elementTags[0]:
    #     # `elementType', `nodeTags', `dim', `tag'
    #     # (4, array([379190, 228334, 237212,   1405], dtype=uint64), 3, 1)
    #     _, nodeTags, _, _ = gmsh.model.mesh.getElement(elementTag)
    #     print(nodeTags)
    #     for nodeTag in nodeTags:
    #         coord4 = coords.reshape(-1, 3)[nodeTag-1]
    #         # print(coord4)
    #         '''
    #         [[27.82440119 28.08032202 64.14131458]
    #          [-3.05059744  3.52157298 35.26255455]
    #          [48.27919376  6.30861458 30.49761989]
    #          [15.69293798 51.73497322 15.73100553]]
    #         '''
    #         # 如果这四个有一个坐标大于position，则将该元素标记为去除
    #         if (coord4[:, 0] < 0).any() and (coord4[:, 1] < 0).any() and (coord4[:, 2] < 0).any():
    #             pass
                
    # exit()
    
    # 写入新的网格文件
    output_file_path = file_path
    gmsh.write(output_file_path)
    gmsh.finalize()

    msh = mesh_io.read_msh(output_file_path)
    mesh_io.write_msh(msh,'TI_via_leadfields.msh')



def performones(leadfield_hdf, mesh_path=None, electrodepair=None):
    startTime = time.time()
    
    leadfield, mesh, idx_lf = TI.load_leadfield(leadfield_hdf=leadfield_hdf)
    
    mout = copy.deepcopy(mesh)

    config_path = 'config_tTIS.json'
    with open(config_path, 'r', encoding='utf-8') as file:
        cfg = json.load(file)
    
    data_path = cfg['data_path']
    workspace = os.path.dirname(data_path)
    m2m_path = os.path.join(workspace, 'm2m_ernie')

    case = cfg['case']

    if case == 1:
        ROIcfg = cfg['case1']
    elif case == 2:
        ROIcfg = cfg['case2']

    ROIcoordMNI = ROIcfg['coordMNI']

    position = simnibs.mni2subject_coords(ROIcoordMNI[0], m2m_path)

    TIpair1 = electrodepair[0]
    TIpair2 = electrodepair[1]
    
    # get fields for the two pairs
    ef1 = TI.get_field_gpu(TIpair1, leadfield, idx_lf)
    ef2 = TI.get_field_gpu(TIpair2, leadfield, idx_lf)

    hlpvar = mesh_io.ElementData(cp.asnumpy(ef1), mesh=mout)
    # mout.add_element_field(hlpvar.norm(),'E_magn1')
    hlpvar = mesh_io.ElementData(cp.asnumpy(ef2), mesh=mout)
    # mout.add_element_field(hlpvar.norm(),'E_magn2')

    TImax = TI.get_maxTI_gpu(ef1, ef2)
    # TImax = normalize_to_range(TImax, new_min=-0.2, new_max=0.2)
    hlpvar = mesh_io.ElementData(cp.asnumpy(TImax), mesh=mout)
    # mout.add_element_field(hlpvar.norm(), 'TImax_ori')
    mout.add_element_field(hlpvar.norm(), 'TImax')

    del ef1, ef2, TImax
    cp._default_memory_pool.free_all_blocks()

    # plot_elec_clip(cfg, electrodepair, TImax, mesh)
    
    rm_list_node = []
    rm_list_elm = []
    for idx, coords in tqdm(enumerate(mout.elm_node_coords())):
        # remove elem in 2nd quadrant
        if (coords[:, 0] > 0).any() and (coords[:, 1] < 0).any() and (coords[:, 2] < 20).any():
            rm_list_elm.append(idx + 1)

    for idx, coord in tqdm(enumerate(mout.nodes.node_coord)):
        if (coord[0] > 0) or (coord[1] < 0) or (coord[2] < 20):
            rm_list_node.append(idx + 1)

    print(f"[performones]before removing {mout.nodes.nr} : {mout.elm.nr}")
    mout = mout.crop_mesh(nodes=rm_list_node, elements=rm_list_elm)
    print(f"[performones]after removing {mout.nodes.nr} : {mout.elm.nr}")

    print(f"[performones]INFO: end:{time.ctime()}, cost: {time.time()-startTime:.4f} seconds")
    mesh_io.write_msh(mout,'TI_via_leadfields.msh')

    # v = mout.view(visible_fields=['TImax_ori', 'TImax'],)
    v = mout.view(visible_fields='TImax',)
    v.write_opt('TI_via_leadfields.msh')

    # cut_model_at_point('TI_via_leadfields.msh', position)

    print(f"[performones]INFO: end:{time.ctime()}, cost: {time.time()-startTime:.4f} seconds")
    mesh_io.open_in_gmsh('TI_via_leadfields.msh', True)
    

if __name__ == "__main__":
    with open('result.json', 'r') as json_file:
        result = json.load(json_file)
    
    TIpair1 = [result['electrodes'][0], result['electrodes'][1], result['currents'][0]]
    TIpair2 = [result['electrodes'][2], result['electrodes'][3], result['currents'][2]]

    performones(leadfield_hdf='D:\ALL\projects\python\simNIBS\examples\leadfield_tet\ernie_leadfield_EEG10-10_UI_Jurak_2007.hdf5', 
                    electrodepair=[TIpair1, TIpair2])
    