#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
this script is used to prepare headmodel and leadfield for tTISE

@Author: boyu
@Date:   2024-9-14 15:29:26
"""
import sys
import os
import glob
import time
import yaml
import json
import shutil
import argparse
from simnibs import sim_struct, run_simnibs
from simnibs.utils.mesh_element_properties import ElementTags



def parse_args():
    parser = argparse.ArgumentParser()
    # windows OS
    parser.add_argument('--data_path', default=".\examples\org", help='Path to where T1 T2 is located')
    parser.add_argument('--subMark', default="ernie", help='number or name')
    parser.add_argument('--output_path', default=".\output", help='Path to output folder, default: ./output')

    return parser.parse_args()


def get_last_two_levels(path=None):
    if path == None: return None
    parts = os.path.split(path)
    # If the path ends with a directory separator, need to split again
    if parts[-1] == '':
        parts = os.path.split(parts[0])
        last_two_levels = os.path.join(os.path.split(parts[-2])[-1], parts[-1])
    else:
        last_two_levels = os.path.join(os.path.split(parts[-2])[-1], parts[-1])
    return last_two_levels


def simNIBS_charm(subMark=None, T1_path=None, T2_path=None):
    start_time = time.time()
    
    # several check
    if subMark == None:
        raise ValueError("subMark must be specified")
    workspace = os.path.dirname(os.path.dirname(T1_path))
    m2m_path = os.path.join(workspace, 'm2m_ernie')
    msh_path = os.path.join(m2m_path, f"{subMark}.msh")
    if os.path.exists(m2m_path):
        if os.path.isfile(msh_path):
            print(f"[simNIBS_charm]INFO: .msh_file already exists: {msh_path} ")
            return
        else:
            shutil.rmtree(m2m_path)
    if T1_path == None:
        raise ValueError("T1_path must be specified")
    
    T1_path = get_last_two_levels(T1_path)
    try:
        # cd to workspace
        os.chdir(workspace)
        print(f"[simNIBS_charm]INFO: current working directory: {os.getcwd()}")
    except OSError as error:
        raise error
    if T2_path:
        T2_path = get_last_two_levels(T2_path)
        cmd = f'charm {subMark} {T1_path} {T2_path}' if T2_path else f'charm {subMark} {T1_path}'
        print(f"[simNIBS_charm]INFO: cmd: {cmd}")
        os.system(cmd)

    print(f"[simNIBS_charm]INFO: simNIBS_charm cost: {time.time()-start_time:.4f}")


def simNIBS_lf_GM(subMark=None, workspace=None):
    '''
    prepare leadfield of gray matter middle surface
    leadfields are defaultly calculated for EEG 10-10 electrode positions
    '''
    start_time = time.time()
    
    # several check
    if subMark == None:
        raise ValueError("subMark must be specified")
    if workspace == None:
        raise ValueError("subject_path must be specified")
    subpath = os.path.join(workspace, 'm2m_ernie')
    msh_file = os.path.join(subpath,f"{subMark}.msh")
    if not os.path.isfile(msh_file):
        raise FileNotFoundError(f"[simNIBS_lf    ]ERROR: .msh_file not found, do charm first")
    leadfield_path = os.path.join(workspace, 'leadfield')
    lf_msh_file = os.path.join(leadfield_path,f'{subMark}_electrodes_EEG10-10_UI_Jurak_2007.msh')
    if os.path.exists(leadfield_path):
        if os.path.isfile(lf_msh_file):
            print(f"[simNIBS_lf]INFO: .msh file already exists: {lf_msh_file} ")
            return
        else:
            shutil.rmtree(leadfield_path)

    tdcs_lf = sim_struct.TDCSLEADFIELD()
    # subject path
    tdcs_lf.subpath = subpath
    # msh file
    tdcs_lf.fnamehead = msh_file
    # output path
    tdcs_lf.pathfem = leadfield_path

    # Uncoment to use the pardiso solver
    tdcs_lf.solver_options = 'pardiso'
    # This solver is faster than the default. However, it requires much more
    # memory (~12 GB)

    run_simnibs(tdcs_lf)
    print(f"[simNIBS_lf]INFO: simNIBS_lf cost: {time.time()-start_time:.4f}")


def simNIBS_lf_GMWM_tet(subMark=None, workspace=None, solver_options=None, map_to_surf=None, tissues=None):
    '''
    prepare leadfield of gray and white matter
    leadfields are defaultly calculated for EEG 10-10 electrode positions
    '''
    start_time = time.time()
    
    # several check
    if subMark == None:
        raise ValueError("subMark must be specified")
    if workspace == None:
        raise ValueError("subject_path must be specified")
    subpath = os.path.join(workspace, 'm2m_ernie')
    msh_file = os.path.join(subpath,f"{subMark}.msh")
    if not os.path.isfile(msh_file):
        raise FileNotFoundError(f"[SIMNIBS_lf_tet]ERROR: .msh_file not found, do charm first")
    tet_path = os.path.join(workspace, 'leadfield_tet')
    lf_msh_file= os.path.join(tet_path,f'{subMark}_electrodes_EEG10-10_UI_Jurak_2007.msh')
    if os.path.exists(tet_path):
        if os.path.isfile(lf_msh_file):
            print(f"[SIMNIBS_lf_tet]INFO: .msh file already exists: {lf_msh_file} ")
            return
        else:
            shutil.rmtree(tet_path)

    tdcs_lf = sim_struct.TDCSLEADFIELD()
    # subject path
    tdcs_lf.subpath = subpath
    # msh file
    tdcs_lf.fnamehead = msh_file
    # output path
    tdcs_lf.pathfem = tet_path

    # Uncoment to use the pardiso solver
    tdcs_lf.solver_options = 'pardiso'
    # This solver is faster than the default. However, it requires much more
    # memory (~12 GB)

    tdcs_lf.interpolation = None
    tdcs_lf.tissues = [ElementTags.GM, ElementTags.WM]
    run_simnibs(tdcs_lf)
    print(f"[SIMNIBS_lf_tet]INFO: SIMNIBS_lf_tet cost: {time.time()-start_time:.4f}")


def prepareHead(cfg=None):
    startTime = time.time()
    print(f"[prepareHead]INFO: start: {time.ctime()}")
    
    # load config if not given
    if not cfg:
        config_path = 'config_tTIS.json'
        with open(config_path, 'r', encoding='utf-8') as file:
            cfg = json.load(file)
    data_path = cfg['data_path']
    subMark = cfg['subMark']
    print(f"[prepareHead]INFO: data_path: {data_path}")
    print(f"[prepareHead]INFO: check T1&T2 -> T1:must | T2:recommended")
    
    # 1. creating head models with charm
    # Check T1
    T1_path = data_path + "\*_T1.nii*"
    T1_path = glob.glob(T1_path)[0]
    if os.path.isfile(T1_path):
        print(f"[prepareHead]INFO: T1_file exists, path:{T1_path}")
    else:
        raise FileNotFoundError(f"[prepareHead   ]ERROR: T1_path not found:{T1_path}")
    
    # Check T2
    T2_path = data_path + "\*_T2.nii*"
    T2_path = glob.glob(T2_path)[0]
    if os.path.isfile(T2_path):
        print(f"[prepareHead]INFO: T2_file exists, path:{T2_path}")
        # run charm with T1 and T2
        simNIBS_charm(subMark, T1_path, T2_path)
    else:
        Warning(f"[prepareHead]WARNING: T2_path not found:{T2_path}")
        # run charm with T1 only
        simNIBS_charm(subMark, T1_path)

    # 2. prepare leadfield of gray matter middle surface
    workspace = os.path.dirname(data_path)
    simNIBS_lf_GM(subMark=subMark, workspace=workspace)

    # 3. prepare leadfield of gray and white matter(tetrahedron)
    simNIBS_lf_GMWM_tet(subMark=subMark, workspace=workspace)

    print(f"[prepareHead]INFO: end:{time.ctime()}, cost: {time.time()-startTime:.4f} seconds")


if __name__ == '__main__':
    prepareHead()
