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
import yaml
import simnibs
from simnibs import mesh_io
import utils.TI_utils as TI
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Rectangle
import scipy.io as sio
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from tqdm import tqdm
from utils.utils import find_positions_by_index, plot_elec, normalize_to_range



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

    max_indices = np.argmax(cp.asnumpy(TImax), axis=0)
    position = find_positions_by_index(mout, 'element', max_indices)[0]
    print(f'position is : {position}')

    del ef1, ef2, TImax
    cp._default_memory_pool.free_all_blocks()

    rm_list_node = []
    rm_list_elm = []

    for idx, coords in enumerate(mout.elm_node_coords()):
        # remove elem in 2nd quadrant
        if (coords[:, 0] > 0).any() and (coords[:, 1] < 0).any() and (coords[:, 2] < 20).any():
            rm_list_elm.append(idx + 1)

    for idx, coord in enumerate(mout.nodes.node_coord):
        if (coord[0] > 0) or (coord[1] < 0) or (coord[2] < 20):
            rm_list_node.append(idx + 1)

    mout = mout.crop_mesh(nodes=rm_list_node, elements=rm_list_elm)

    print(f"[performones]INFO: end:{time.ctime()}, cost: {time.time()-startTime:.4f} seconds")
    mesh_io.write_msh(mout,'TI_via_leadfields.msh')

    v = mout.view(visible_fields='TImax',)
    v.write_opt('TI_via_leadfields.msh')

    print(f"[performones]INFO: end:{time.ctime()}, cost: {time.time()-startTime:.4f} seconds")
    mesh_io.open_in_gmsh('TI_via_leadfields.msh', True)
    

if __name__ == "__main__":
    with open('result.json', 'r') as json_file:
        result = json.load(json_file)
    
    TIpair1 = [result['electrodes'][0], result['electrodes'][1], result['currents'][0]]
    TIpair2 = [result['electrodes'][2], result['electrodes'][3], result['currents'][2]]

    performones(leadfield_hdf='D:\ALL\projects\python\simNIBS\examples\leadfield_tet\ernie_leadfield_EEG10-10_UI_Jurak_2007.hdf5', 
                    electrodepair=[TIpair1, TIpair2])
    