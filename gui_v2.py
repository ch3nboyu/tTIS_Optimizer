#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: boyu
@Date:   2025-12-11 20:49:30
"""
import os.path
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QTableWidgetItem,QHeaderView
from PyQt5.QtCore import Qt, QCoreApplication, QThread, pyqtSignal
from win_ui import Ui_MainWindow
import subprocess
import argparse, yaml
import time, os, glob, shutil
import chardet
import os
import ast
import json
import time, copy
import numpy as np
import gmsh
import yaml
import simnibs
from simnibs import mesh_io
import utils.TI_utils as TI
import argparse, yaml
import time, os, glob, shutil
from simnibs import sim_struct, run_simnibs
from simnibs.utils.mesh_element_properties import ElementTags

# 线程池
class Thread_prepare_headandLF(QThread):
    # 定义一个自定义信号，用于向主线程发送消息
    thread_signal = pyqtSignal(str)
    progressBar_signal = pyqtSignal(int)
    def __init__(self):
        super().__init__()

    def run(self):
        # 进度条
        self.progressBar_signal.emit(0)
        self.prepareHead()
        # 进度条
        self.progressBar_signal.emit(100)

    def detect_encoding(self,file_path):
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            return result['encoding']

    def parse_args(self):
        parser = argparse.ArgumentParser()
        # windows OS
        parser.add_argument('--data_path', default="D:\ALL\projects\python\simNIBS\examples\org", help='Path to where T1 T2 is located')
        parser.add_argument('--subMark', default="ernie", help='number or name')
        parser.add_argument('--output_path', default="D:\ALL\projects\python\simNIBS\output", help='Path to output folder, default: ./output')


        return parser.parse_args()

    def get_last_two_levels(self,path=None):
        if path == None: return None
        parts = os.path.split(path)
        # If the path ends with a directory separator, need to split again
        if parts[-1] == '':
            parts = os.path.split(parts[0])
            last_two_levels = os.path.join(os.path.split(parts[-2])[-1], parts[-1])
        else:
            last_two_levels = os.path.join(os.path.split(parts[-2])[-1], parts[-1])
        return last_two_levels

    def simNIBS_charm(self,subMark=None, T1_path=None, T2_path=None):
        start_time = time.time()

        # several check
        if subMark == None:
            raise ValueError("subMark must be specified")
        workspace = os.path.dirname(os.path.dirname(T1_path))
        m2m_path = os.path.join(workspace, 'm2m_ernie')
        msh_path = os.path.join(m2m_path, f"{subMark}.msh")
        if os.path.exists(m2m_path):
            if os.path.isfile(msh_path):
                self.thread_signal.emit(f"[simNIBS_charm ]INFO: .msh_file already exists: {msh_path} ")
                return
            else:
                shutil.rmtree(m2m_path)
        if T1_path == None:
            raise ValueError("T1_path must be specified")

        T1_path = self.get_last_two_levels(T1_path)
        try:
            # cd to workspace
            os.chdir(workspace)
            self.thread_signal.emit(f"[simNIBS_charm ]INFO: current working directory: {os.getcwd()}")
        except OSError as error:
            raise error
        if T2_path:
            T2_path = self.get_last_two_levels(T2_path)
            cmd = f'charm {subMark} {T1_path} {T2_path}' if T2_path else f'charm {subMark} {T1_path}'
            self.thread_signal.emit(f"[simNIBS_charm ]INFO: cmd: {cmd}")
            os.system(cmd)

        self.thread_signal.emit(f"[simNIBS_charm ]INFO: simNIBS_charm cost: {time.time() - start_time:.4f}")

    def simNIBS_lf_GM(self,subMark=None, workspace=None):
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
        msh_file = os.path.join(subpath, f"{subMark}.msh")
        if not os.path.isfile(msh_file):
            raise FileNotFoundError(f"[simNIBS_lf    ]ERROR: .msh_file not found, do charm first")
        leadfield_path = os.path.join(workspace, 'leadfield')
        lf_msh_file = os.path.join(leadfield_path, f'{subMark}_electrodes_EEG10-10_UI_Jurak_2007.msh')
        if os.path.exists(leadfield_path):
            if os.path.isfile(lf_msh_file):
                self.thread_signal.emit(f"[simNIBS_lf    ]INFO: .msh file already exists: {lf_msh_file} ")
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

        self.thread_signal.emit(f"[simNIBS_lf    ]INFO: simNIBS_lf cost: {time.time() - start_time:.4f}")

    def simNIBS_lf_GMWM_tet(self,subMark=None, workspace=None, solver_options=None, map_to_surf=None, tissues=None):
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
        msh_file = os.path.join(subpath, f"{subMark}.msh")
        if not os.path.isfile(msh_file):
            raise FileNotFoundError(f"[SIMNIBS_lf_tet]ERROR: .msh_file not found, do charm first")
        tet_path = os.path.join(workspace, 'leadfield_tet')
        lf_msh_file = os.path.join(tet_path, f'{subMark}_electrodes_EEG10-10_UI_Jurak_2007.msh')
        if os.path.exists(tet_path):
            if os.path.isfile(lf_msh_file):
                self.thread_signal.emit(f"[SIMNIBS_lf_tet]INFO: .msh file already exists: {lf_msh_file} ")
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

        # tdcs_lf.interpolation = None
        tdcs_lf.tissues = [ElementTags.WM_TH_SURFACE, ElementTags.GM_TH_SURFACE]
        run_simnibs(tdcs_lf)
        self.thread_signal.emit(f"[SIMNIBS_lf_tet]INFO: SIMNIBS_lf_tet cost: {time.time() - start_time:.4f}")

    def prepareHead(self,cfg=None):
        startTime = time.time()
        self.thread_signal.emit(f"[prepareHead   ]INFO: start: {time.ctime()}")

        # load config if not given
        if not cfg:
            config_path = 'config_tTIS.yaml'
            with open(config_path, 'r', encoding=self.detect_encoding(config_path)) as file:
                cfg = yaml.safe_load(file)
        data_path = cfg['data_path']
        subMark = cfg['subMark']
        self.thread_signal.emit(f"[prepareHead   ]INFO: data_path: {data_path}")
        self.thread_signal.emit(f"[prepareHead   ]INFO: check T1&T2 -> T1:must | T2:recommended")
        # 更新进度
        self.progressBar_signal.emit(10)
        # 1. creating head models with charm
        # Check T1
        T1_path = data_path + "\*_T1.nii*"
        T1_path = glob.glob(T1_path)[0]
        if os.path.isfile(T1_path):
            self.thread_signal.emit(f"[prepareHead   ]INFO: T1_file exists, path:{T1_path}")
        else:
            raise FileNotFoundError(f"[prepareHead   ]ERROR: T1_path not found:{T1_path}")
        # 更新进度
        self.progressBar_signal.emit(15)
        # Check T2
        T2_path = data_path + "\*_T2.nii*"
        T2_path = glob.glob(T2_path)[0]
        if os.path.isfile(T2_path):
            self.thread_signal.emit(f"[prepareHead   ]INFO: T2_file exists, path:{T2_path}")
            # run charm with T1 and T2
            self.simNIBS_charm(subMark, T1_path, T2_path)
        else:
            Warning(f"[prepareHead   ]WARNING: T2_path not found:{T2_path}")
            # run charm with T1 only
            self.simNIBS_charm(subMark, T1_path)
        # 更新进度
        self.progressBar_signal.emit(20)
        # 2. prepare leadfield of gray matter middle surface
        workspace = os.path.dirname(data_path)
        self.simNIBS_lf_GM(subMark=subMark, workspace=workspace)
        # 更新进度
        self.progressBar_signal.emit(25)
        # 3. prepare leadfield of gray and white matter
        self.simNIBS_lf_GMWM_tet(subMark=subMark, workspace=workspace)
        # 更新进度
        self.progressBar_signal.emit(30)
        self.thread_signal.emit(f"[prepareHead   ]INFO: end:{time.ctime()}, cost: {time.time() - startTime:.4f} seconds")


# 线程池
class Thread_performones_tTIS(QThread):
    # 定义一个自定义信号，用于向主线程发送消息
    thread_signal = pyqtSignal(str)
    progressBar_signal = pyqtSignal(int)
    def __init__(self):
        super().__init__()

    def run(self):
        # 进度条
        self.progressBar_signal.emit(0)
        # load result
        with open('result.json', 'r', encoding=self.detect_encoding('result.json')) as json_file:
            result = json.load(json_file)

        TIpair1 = [result['electrodes'][0], result['electrodes'][1], result['currents'][0]]
        TIpair2 = [result['electrodes'][2], result['electrodes'][3], result['currents'][2]]

        self.performones(leadfield_hdf='examples\leadfield_tet\ernie_leadfield_EEG10-10_UI_Jurak_2007.hdf5',
                    electrodepair=[TIpair1, TIpair2])
        # 进度条
        self.progressBar_signal.emit(100)

    def detect_encoding(self,file_path):
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            return result['encoding']


    def read_and_save_msh(self,input_file_path, output_file_path):


        if input_file_path.endswith('.geo'):
            gmsh.open(input_file_path)
            gmsh.model.geo.synchronize()
        elif input_file_path.endswith('.msh'):
            gmsh.merge(input_file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a .geo or .msh file.")

        gmsh.write(output_file_path)

        gmsh.finalize()


    def performones(self,leadfield_hdf, mesh_path=None, electrodepair=None):
        startTime = time.time()

        leadfield, mesh, idx_lf = TI.load_leadfield(leadfield_hdf=leadfield_hdf)

        # 253140 个node
        mout = copy.deepcopy(mesh)
        # 进度条
        self.progressBar_signal.emit(20)
        config_path = 'config_tTIS.yaml'
        with open(config_path, 'r', encoding=self.detect_encoding(config_path)) as file:
            cfg = yaml.safe_load(file)

        data_path = cfg['data_path']
        workspace = os.path.dirname(data_path)
        m2m_path = os.path.join(workspace, 'm2m_ernie')

        case = cfg['case']

        if case == 1:
            ROIcfg = cfg['case1']
        elif case == 2:
            ROIcfg = cfg['case2']

        ROIcoordMNI = ROIcfg['coordMNI']
        positions = simnibs.mni2subject_coords(ROIcoordMNI[0], m2m_path)

        TIpair1 = electrodepair[0]
        TIpair2 = electrodepair[1]

        # get fields for the two pairs
        ef1 = TI.get_field(TIpair1, leadfield, idx_lf)
        ef2 = TI.get_field(TIpair2, leadfield, idx_lf)
        # 进度条
        self.progressBar_signal.emit(50)

        TImax = TI.get_maxTI(ef1, ef2)
        mout.add_element_field(TImax, 'TImax')  # for visualization

        keep_nodes = []
        for idx, cord in enumerate(mout.nodes.node_coord):
            if cord[0] < 0 or cord[1] < 0 or cord[2] < 0:
                keep_nodes.append(idx)
        mout = mout.crop_mesh(nodes=keep_nodes)

        mesh_io.write_msh(mout, 'TI_via_leadfields.msh')
        # 进度条
        self.progressBar_signal.emit(70)
        v = mout.view(
            visible_tags=[1, 2],
            visible_fields='E_magn1',
        )

        v.write_opt('TI_via_leadfields.msh')

        self.read_and_save_msh('TI_via_leadfields.msh', 'cut_TI_via_leadfields.msh')
        # 进度条
        self.progressBar_signal.emit(80)
        mesh_io.open_in_gmsh('cut_TI_via_leadfields.msh', True)

        self.thread_signal.emit(f"[performones    ]INFO: end:{time.ctime()}, cost: {time.time() - startTime:.4f} seconds")

class Thread_optimization_tTIS(QThread):
    # 定义一个自定义信号，用于向主线程发送消息
    thread_signal = pyqtSignal(str)
    progressBar_signal = pyqtSignal(int)


    def __init__(self):
        super().__init__()

    def run(self):
        self.thread_signal.emit("暂未开发，以下为测试程序")
        num = 0
        self.progressBar_signal.emit(int(num))
        while True:
            self.thread_signal.emit(str(num))
            time.sleep(1)
            num += 1
            # 进度条
            self.progressBar_signal.emit(int(num*10))
            if num == 10:
                break
        self.thread_signal.emit("结束")
        self.progressBar_signal.emit(100)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):  # 构造方法
        super(MainWindow, self).__init__()  # 运行父类的构造方法
        self.setupUi(self)
        # 初始化配置
        self.ini_config()
        self.save_button.clicked.connect(self.save_config)
        self.prepare_button.clicked.connect(self.prepare_headandLF)
        self.optimization_button.clicked.connect(self.optimization_tTIS)
        self.simulation_button.clicked.connect(self.performones_tTIS)


    def update_progressBar(self,value):
        self.progressBar.setValue(value)
        if value >= 100:
            self.save_button.setEnabled(True)
            self.prepare_button.setEnabled(True)
            self.optimization_button.setEnabled(True)
            self.simulation_button.setEnabled(True)

    def detect_encoding(self,file_path):
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            return result['encoding']


    # 初始化配置
    def ini_config(self):
        config_path = 'config_tTIS.json'
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding=self.detect_encoding(config_path)) as file:
                cfg = json.load(file)
                
                data_path = cfg['data_path']
                self.data_path_edit.setText(str(data_path))
                output_path = cfg['output_path']
                self.output_path_edit.setText(str(output_path))
                subMark = cfg['subMark']
                self.subMark_edit.setText(str(subMark))
                simMark = cfg['simMark']
                self.simMark_edit.setText(str(simMark))
                exampleIdx = cfg['exampleIdx']
                self.exampleIdx_edit.setText(str(exampleIdx))
                eeg_posi = cfg['eeg-posi']
                self.eeg_posi_edit.setText(str(eeg_posi))
                Element_Type = cfg['type']
                self.type_edit.setCurrentText(str(Element_Type))
                thres = cfg['thres']
                self.thres_edit.setText(str(thres))
                initialElecNum = cfg['initialElecNum']
                self.initialElecNum_edit.setText(str(initialElecNum))
                cur_max = cfg['cur_max']
                self.cur_max_edit.setText(str(cur_max))
                cur_min = cfg['cur_min']
                self.cur_min_edit.setText(str(cur_min))
                cur_sum = cfg['cur_sum']
                self.cur_sum_edit.setText(str(cur_sum))
                precision = cfg['precision']
                self.precision_edit.setText(str(precision))
                nt = cfg['nt']
                self.nt_edit.setText(str(nt))
                Method = cfg['Method']
                self.method_combo.setCurrentText(str(Method))
                max_iter = cfg['GA_para']['max_iter']
                self.max_iter_edit.setText(str(max_iter))
                pop_size = cfg['GA_para']['pop_size']
                self.pop_size_edit.setText(str(pop_size))
                mut_prob = cfg['GA_para']['mut_prob']
                self.mut_prob_edit.setText(str(mut_prob))
                elit_ratio = cfg['GA_para']['elit_ratio']
                self.elit_ratio_edit.setText(str(elit_ratio))
                cross_prob = cfg['GA_para']['cross_prob']
                self.cross_prob_edit.setText(str(cross_prob))
                parents_portion = cfg['GA_para']['parents_portion']
                self.parents_portion_edit.setText(str(parents_portion))
                # cross_type = cfg['GA_para']['cross_type']
                # self.cross_type_edit.setText(str(cross_type))
                # time_threshold = cfg['GA_para']['time_threshold']
                # self.time_threshold_edit.setText(str(time_threshold))
                para_1 = cfg['PSO_para']['para_1']
                self.para_1_edit.setText(str(para_1))
                para_2 = cfg['PSO_para']['para_2']
                self.para_2_edit.setText(str(para_2))
                para_3 = cfg['PSO_para']['para_3']
                self.para_3_edit.setText(str(para_3))
                para_4 = cfg['PSO_para']['para_4']
                self.para_4_edit.setText(str(para_4))
                para_5 = cfg['PSO_para']['para_5']
                self.para_5_edit.setText(str(para_5))
                num_case1 = cfg['case1']['num']
                self.case1_num_edit.setText(str(num_case1))
                name_case1 = cfg['case1']['name']
                self.case1_name_edit.setText(str(name_case1))
                coordMNI_case1 = cfg['case1']['coordMNI']
                self.case1_coordMNI_edit.setText(str(coordMNI_case1))
                shape_case1 = cfg['case1']['shape']
                self.case1_shape_edit.setText(str(shape_case1))
                radius_case1 = cfg['case1']['radius']
                self.case1_radius_edit.setText(str(radius_case1))
                intensity_case1 = cfg['case1']['intensity']
                self.case1_intensity_edit.setText(str(intensity_case1))
                otheralpha_case1 = cfg['case1']['otheralpha']
                self.case1_otheralpha_edit.setText(str(otheralpha_case1))
                num_case2 = cfg['case2']['num']
                self.case2_num_edit.setText(str(num_case2))
                name_case2 = cfg['case2']['name']
                self.case2_name_edit.setText(str(name_case2))
                coordMNI_case2 = cfg['case2']['coordMNI']
                self.case2_coordMNI_edit.setText(str(coordMNI_case2))
                shape_case2 = cfg['case2']['shape']
                self.case2_shape_edit.setText(str(shape_case2))
                radius_case2 = cfg['case2']['radius']
                self.case2_radius_edit.setText(str(radius_case2))
                intensity_case2 = cfg['case2']['intensity']
                self.case2_intensity_edit.setText(str(intensity_case2))
                otheralpha_case2 = cfg['case2']['otheralpha']
                self.case2_otheralpha_edit.setText(str(otheralpha_case2))
                avoid_num = cfg['avoid']['num']
                self.avoid_num_edit.setText(str(avoid_num))
                avoid_name = cfg['avoid']['name']
                self.avoid_name_edit.setText(str(avoid_name))
                avoid_coordMNI = cfg['avoid']['coordMNI']
                self.avoid_coordMNI_edit.setText(str(avoid_coordMNI))
                avoid_shape = cfg['avoid']['shape']
                self.avoid_shape_edit.setText(str(avoid_shape))
                avoid_radius = cfg['avoid']['radius']
                self.avoid_radius_edit.setText(str(avoid_radius))
                avoid_coef = cfg['avoid']['coef']
                self.avoid_coef_edit.setText(str(avoid_coef))


    def save_config(self):
        config = {
            "data_path": self.data_path_edit.text(),
            "output_path": self.output_path_edit.text(),
            "subMark": self.subMark_edit.text(),
            "simMark": self.simMark_edit.text(),
            "exampleIdx": int(self.exampleIdx_edit.text()),
            "eeg-posi": self.eeg_posi_edit.text(),
            "initialElecNum": int(self.initialElecNum_edit.text()),
            "cur_max": float(self.cur_max_edit.text()),
            "cur_min": float(self.cur_min_edit.text()),
            "cur_sum": float(self.cur_sum_edit.text()),
            "precision": float(self.precision_edit.text()),
            "type": self.type_edit.currentText(),
            "thres": float(self.thres_edit.text()),
            "nt": int(self.nt_edit.text()),
            "Method": self.method_combo.currentText(),
            "GA_para": {
                "max_iter": int(self.max_iter_edit.text()),
                "pop_size": int(self.pop_size_edit.text()),
                "mut_prob": float(self.mut_prob_edit.text()),
                "elit_ratio": float(self.elit_ratio_edit.text()),
                "cross_prob": float(self.cross_prob_edit.text()),
                "parents_portion": float(self.parents_portion_edit.text())
            },

            "PSO_para": {
                "para_1": int(self.para_1_edit.text()),
                "para_2": float(self.para_2_edit.text()),
                "para_3": float(self.para_3_edit.text()),
                "para_4": float(self.para_4_edit.text()),
                "para_5": float(self.para_5_edit.text())
            },
            "case":int(self.tabWidget_case.currentIndex())+1,
            "case1": {
                "num": int(self.case1_num_edit.text()),
                "name": ast.literal_eval(self.case1_name_edit.text()),
                "coordMNI": ast.literal_eval(self.case1_coordMNI_edit.text()),
                "shape": ast.literal_eval(self.case1_shape_edit.text()),
                "radius": ast.literal_eval(self.case1_radius_edit.text()),
                "intensity": float(self.case1_intensity_edit.text()),
                "otheralpha": float(self.case1_otheralpha_edit.text()),
            },
            "case2": {
                "num": int(self.case2_num_edit.text()),
                "name": ast.literal_eval(self.case2_name_edit.text()),
                "coordMNI": ast.literal_eval(self.case2_coordMNI_edit.text()),
                "shape": ast.literal_eval(self.case2_shape_edit.text()),
                "radius": ast.literal_eval(self.case2_radius_edit.text()),
                "intensity": float(self.case2_intensity_edit.text()),
                "otheralpha": float(self.case2_otheralpha_edit.text()),
            },
            "avoid": {
                "num": int(self.avoid_num_edit.text()),
                "name": ast.literal_eval(self.avoid_name_edit.text()),
                "coordMNI": ast.literal_eval(self.avoid_coordMNI_edit.text()),
                "shape": ast.literal_eval(self.avoid_shape_edit.text()),
                "radius": ast.literal_eval(self.avoid_radius_edit.text()),
                "coef": float(self.avoid_coef_edit.text()),

            },
        }
        with open('config_tTIS_test.yaml', 'w') as file:
            json.dump(config, file, indent=4)
        self.display_message('cfg saved')

    def display_message(self, message):
        self.right_widget.append(message)


    def prepare_headandLF(self):
        self.prepare_button.setEnabled(False)
        self.optimization_button.setEnabled(False)
        self.simulation_button.setEnabled(False)
        self.save_button.setEnabled(False)

        self.display_message('Running prepare_headandLF.py...')
        self.thread = Thread_prepare_headandLF()
        self.thread.thread_signal.connect(self.msg_task)
        self.thread.progressBar_signal.connect(self.update_progressBar)
        self.thread.start()


    def optimization_tTIS(self):
        self.prepare_button.setEnabled(False)
        self.optimization_button.setEnabled(False)
        self.simulation_button.setEnabled(False)
        self.save_button.setEnabled(False)

        self.display_message('Running optimization_tTIS.py...')
        self.thread = Thread_optimization_tTIS()
        self.thread.thread_signal.connect(self.msg_task)
        self.thread.progressBar_signal.connect(self.update_progressBar)
        self.thread.start()

    def performones_tTIS(self):
        self.prepare_button.setEnabled(False)
        self.optimization_button.setEnabled(False)
        self.simulation_button.setEnabled(False)
        self.save_button.setEnabled(False)

        self.display_message('Running performones_tTIS.py...')
        self.thread = Thread_performones_tTIS()
        self.thread.thread_signal.connect(self.msg_task)
        self.thread.progressBar_signal.connect(self.update_progressBar)
        self.thread.start()


    def msg_task(self,msg):
        self.display_message(msg)


if __name__ == '__main__':
    # 分辨率自适应
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    # 适应高DPI设备
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    # 解决图片在不同分辨率显示模糊问题
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
