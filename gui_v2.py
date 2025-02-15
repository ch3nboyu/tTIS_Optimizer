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
import chardet
import os
import ast
import json, h5py
import numpy as np
import gmsh
import simnibs
from simnibs import mesh_io
import utils.TI_utils as TI
import argparse, yaml
import time, os, glob, shutil
from simnibs import sim_struct, run_simnibs
from simnibs.utils.mesh_element_properties import ElementTags
import re
import csv
import copy
import scipy
import math
import logging
from tqdm import tqdm
from scipy.optimize import minimize
from sko.PSO import PSO
import cupy as cp
import matplotlib.pyplot as plt
from simnibs.mesh_tools import mesh_io, gmsh_view
from simnibs.utils.simnibs_logger import logger
from simnibs.optimization import optimization_methods
from utils.GA import objective_df
from geneticalgorithm import geneticalgorithm as ga
from utils.utils import readMatFile, plot_elec





def _find_indexes(mesh, lf_type, indexes=None, positions=None, tissues=None, radius=0.):
    ''' 
    在网格中查找满足以下条件之一的节点/元素：
    1. 在指定组织内，且位于一组给定点（由positions定义）的指定半径范围内的节点/元素。
       第一步是找到最近的节点/元素。
    2. 特定的索引

    返回网格中节点/元素的索引，以及一个映射，表明新点是从原始点中的哪个点获取的。
    '''
    # 检查是否同时定义了positions和indexes，或者两者都未定义。如果是，则抛出错误。
    if (positions is not None) == (indexes is not None): # 负XOR操作
        raise ValueError('请定义positions或indexes，但不能同时定义两者')

    # 如果indexes不为None，则直接返回indexes及其对应的映射。
    if indexes is not None:
        indexes = np.atleast_1d(indexes)  # 确保indexes是一个数组
        return indexes, np.arange(len(indexes))

    # 根据lf_type的类型（'node'或'element'）来获取相应的网格索引和位置。
    if lf_type == 'node':
        # 如果指定了tissues，则获取具有这些标签的节点索引，否则获取所有节点索引。
        if tissues is not None:
            mesh_indexes = mesh.elm.nodes_with_tag(tissues)
        else:
            mesh_indexes = mesh.nodes.node_number

        # 获取这些节点的位置。
        mesh_pos = mesh.nodes[mesh_indexes]

    elif lf_type == 'element':
        # 如果指定了tissues，则获取具有这些标签的元素索引，否则获取所有元素索引。
        if tissues is not None:
            mesh_indexes = mesh.elm.elm_number[np.isin(mesh.elm.tag1, tissues)]
        else:
            mesh_indexes = mesh.elm.elm_number

        # 获取这些元素的质心位置。
        mesh_pos = mesh.elements_baricenters()[mesh_indexes]

    else:
        # 如果lf_type不是'node'或'element'，则抛出错误。
        raise ValueError('lf_type必须是"node"或"element"')

    # 确保radius为非负数，并且mesh_pos不为空。
    assert radius >= 0., 'radius应 >= 0'
    assert len(mesh_pos) > 0, '未找到具有给定标签的任何元素或节点'

    # 使用KDTree来快速查找最近的点。
    kdtree = scipy.spatial.cKDTree(mesh_pos)
    pos_projected, indexes = kdtree.query(positions)
    indexes = np.atleast_1d(indexes)

    # 如果radius大于一个很小的值（1e-9），则查找在指定半径内的所有点。
    if radius > 1e-9:
        in_radius = kdtree.query_ball_point(mesh_pos[indexes], radius)
        # 创建一个映射，表明每个新点是从原始点中的哪个点获取的。
        original = np.concatenate([(i,)*len(ir) for i, ir in enumerate(in_radius)])
        # 去除重复的点，并返回唯一的索引和映射
        in_radius, uq_idx = np.unique(np.concatenate(in_radius), return_index=True)
        return mesh_indexes[in_radius], original[uq_idx]
    else:
        # 如果radius很小或为0，则直接返回最近的点的索引和映射。
        return mesh_indexes[indexes],  np.arange(len(indexes))


def _find_directions(mesh, lf_type, directions, indexes, mapping=None):
    if directions is None:
        return None

    if directions == 'normal':
        if 4 in np.unique(mesh.elm.elm_type):
            return None
            # raise ValueError("Can't define a normal direction for volumetric data!")

        if lf_type == 'node':
            directions = -mesh.nodes_normals()[indexes]
        elif lf_type == 'element':
            directions = -mesh.triangle_normals()[indexes]
        return directions

    else:
        directions = np.atleast_2d(directions)

        if directions.shape[1] != 3:
            raise ValueError(
                "directions must be the string 'normal' or a Nx3 array"
            )

        if mapping is None:
            if len(directions) == len(indexes):
                mapping = np.arange(len(indexes))
            else:
                raise ValueError('Different number of indexes and directions and no '
                                 'mapping defined')

        elif len(directions) == 1:
            mapping = np.zeros(len(indexes), dtype=int)

        directions = directions/np.linalg.norm(directions, axis=1)[:, None]

        return directions[mapping]


class tTIS_opt():
    def __init__(self, leadfield_hdf=None,
                 max_total_current=2.0,
                 max_individual_current=1.5,
                 min_individual_current=0.5,
                 precision=0.01,
                 active_electrodes=4,
                 name='optimization',
                 nt=False,
                 target=None,
                 avoid=None,
                 opt_method=None,
                 method_para=None,
                 open_in_gmsh=True):
        
        self.leadfield_hdf = leadfield_hdf
        self.max_total_current = max_total_current
        self.max_individual_current = max_individual_current
        self.min_individual_current = min_individual_current
        self.precision = precision
        self.active_electrodes = active_electrodes
        self.nt = nt
        self.leadfield_path = '/mesh_leadfield/leadfields/tdcs_leadfield'
        self.mesh_path = '/mesh_leadfield/'
        self.opt_method = opt_method
        self.method_para = method_para
        self.open_in_gmsh = open_in_gmsh
        self._mesh = None
        self._leadfield = None
        self._field_name = None
        self._field_units = None
        self.name = name
        # I can't put [] in the arguments for weird reasons (it gets the previous value)
        if opt_method is None:
            self.opt_method = {}
        if target is None:
            self.target = []
        else:
            self.target = target
        if avoid is None:
            self.avoid = []
        else:
            self.avoid = avoid


    @property
    def lf_type(self):
        if self.mesh is None or self.leadfield is None:
            return None
        if self.leadfield.shape[1] == self.mesh.nodes.nr:
            return 'node'
        elif self.leadfield.shape[1] == self.mesh.elm.nr:
            return 'element'
        else:
            raise ValueError('Could not find if the leadfield is node- or '
                              'element-based')
        
    @property
    def leadfield(self):
        ''' Reads the leadfield from the HDF5 file'''
        if self._leadfield is None and self.leadfield_hdf is not None:
            with h5py.File(self.leadfield_hdf, 'r') as f:
                self.leadfield = f[self.leadfield_path][:]
        return self._leadfield
    
    @leadfield.setter
    def leadfield(self, leadfield):
        if leadfield is not None:
            assert leadfield.ndim == 3, 'leadfield should be 3 dimensional'
            assert leadfield.shape[2] == 3, 'Size of last dimension of leadfield should be 3'
        self._leadfield = leadfield

    @property
    def mesh(self):
        if self._mesh is None and self.leadfield_hdf is not None:
            self.mesh = mesh_io.Msh.read_hdf5(self.leadfield_hdf, self.mesh_path)
        return self._mesh
    
    @mesh.setter
    def mesh(self, mesh):
        elm_type = np.unique(mesh.elm.elm_type)
        if len(elm_type) > 1:
            raise ValueError('Mesh has both tetrahedra and triangles')
        else:
            self._mesh = mesh

    @property
    def field_name(self):
        if self.leadfield_hdf is not None and self._field_name is None:
            try:
                with h5py.File(self.leadfield_hdf, 'r') as f:
                    self.field_name = f[self.leadfield_path].attrs['field']
            except:
                return 'Field'
        if self._field_name is None:
            return 'Field'
        else:
            return self._field_name
    
    @field_name.setter
    def field_name(self, field_name):
        self._field_name = field_name

    @property
    def field_units(self):
        if self.leadfield_hdf is not None and self._field_units is None:
            try:
                with h5py.File(self.leadfield_hdf, 'r') as f:
                    self.field_units = f[self.leadfield_path].attrs['units']
            except:
                return 'Au'

        if self._field_units is None:
            return 'Au'
        else:
            return self._field_units

    @field_units.setter
    def field_units(self, field_units):
        self._field_units = field_units

    def mesh_index(self, tissues=None):
        """
        根据lf_type和给定的组织标签，返回网格索引和位置。

        Args:
            tissues (list, optional): 组织标签列表。默认为None。

        Returns:
            tuple: 包含两个元素的元组。
                - mesh_indexes (np.ndarray): 网格索引数组。
                - mesh_pos (np.ndarray): 网格位置数组。

            Raises:
                ValueError: 如果lf_type既不是"node"也不是"element"时抛出。
        """
        if self.lf_type == 'node':
            if tissues is not None:
                mesh_indexes = self.mesh.elm.nodes_with_tag(tissues)
            else:
                mesh_indexes = self.mesh.nodes.node_number

            mesh_pos = self.mesh.nodes[mesh_indexes]

        elif self.lf_type == 'element':
            if tissues is not None:
                mesh_indexes = self.mesh.elm.elm_number[np.isin(self.mesh.elm.tag1, tissues)]
            else:
                mesh_indexes = self.mesh.elm.elm_number

            mesh_pos = self.mesh.elements_baricenters()[mesh_indexes]

        else:
            raise ValueError('lf_type must be either "node" or "element"')
        
        return mesh_indexes, mesh_pos


        
    
    def get_weights(self):
        ''' Calculates the volumes or areas of the mesh associated with the leadfield
        '''
        assert self.mesh is not None, 'Mesh not defined'
        if self.lf_type == 'node':
            weights = self.mesh.nodes_volumes_or_areas().value
        elif self.lf_type == 'element':
            weights = self.mesh.elements_volumes_and_areas().value
        else:
            raise ValueError('Cant calculate weights: mesh or leadfield not set')

        weights *= self._get_avoid_field()
        return weights

    def _get_avoid_field(self):
        fields = []
        for a in self.avoid:
            a.mesh = self.mesh
            a.lf_type = self.lf_type
            fields.append(a.avoid_field())

        if len(fields) > 0:
            total_field = np.ones_like(fields[0])
            for f in fields:
                total_field *= f
            return total_field

        else:
            return 1.

    def add_target(self, target=None):
        ''' Adds a target to the current tDCS optimization

        Parameters:
        ------------
        target: tTIStarget (optional)
            TDCStarget structure to be added. Default: empty TDCStarget

        Returns:
        -----------
        target: tTIStarget
            tTIStarget added to the structure
        '''
        if target is None:
            target = tTIStarget(mesh=self.mesh, lf_type=self.lf_type)
        self.target.append(target)
        return target

    def add_avoid(self, avoid=None):
        ''' Adds an avoid structure to the current tDCS optimization

        Parameters:
        ------------
        target: TDCStarget (optional)
            TDCStarget structure to be added. Default: empty TDCStarget

        Returns:
        -----------
        target: TDCStarget
            TDCStarget added to the structure
        '''
        if avoid is None:
            avoid = tTISavoid(mesh=self.mesh, lf_type=self.lf_type)
        self.avoid.append(avoid)
        return avoid

    def _assign_mesh_lf_type_to_target(self):
        for t in self.target:
            if t.mesh is None: t.mesh = self.mesh
            if t.lf_type is None: t.lf_type = self.lf_type
        for a in self.avoid:
            if a.mesh is None: a.mesh = self.mesh
            if a.lf_type is None: a.lf_type = self.lf_type

    def optimize(self, fn_out_mesh=None, fn_out_csv=None):
        startTime = time.time()
        ''' Runs the optimization problem
        '''
        assert len(self.target) > 0, 'No target defined'
        assert self.leadfield is not None, 'Leadfield not defined'
        assert self.mesh is not None, 'Mesh not defined'
        if self.active_electrodes is not None:
            assert self.active_electrodes > 3, \
                    'The number of active electrodes should be at least 4'

        if self.max_total_current is None:
            logger.warning('Maximum total current not set!')
            max_total_current = 2.0

        else:
            assert self.max_total_current > 0
            max_total_current = self.max_total_current

        if self.max_individual_current is None:
            max_individual_current = max_total_current

        else:
            assert self.max_individual_current > 0
            max_individual_current = self.max_individual_current

        method_para = self.method_para
        
        max_iter = method_para['max_iter']
        pop_size = method_para['pop_size']
        mut_prob = method_para['mut_prob']
        elit_ratio = method_para['elit_ratio']
        cross_prob = method_para['cross_prob']
        parents_portion = method_para['parents_portion']
        cross_type = 'uniform'
        time_threshold = 120

        algorithm_param = {'max_num_iteration': max_iter,
                        'population_size': pop_size,
                        'mutation_probability': mut_prob,
                        'elit_ratio': elit_ratio,
                        'crossover_probability': cross_prob,
                        'parents_portion': parents_portion,
                        'crossover_type': cross_type,
                        'max_iteration_without_improv': time_threshold
                        }
        
        self._assign_mesh_lf_type_to_target()
        weights = self.get_weights()
        index, pos = self.mesh_index()
        leadfield = self.leadfield

        roi_ids_list = []
        avoid_ids_list = []

        for t in self.target:
            roi_ids = t.get_indexes()
            roi_ids_list.append(roi_ids)
        roi_ids = np.concatenate(roi_ids_list)
        
        for a in self.avoid:
            avoid_ids = a.get_indexes()
            avoid_ids_list.append(avoid_ids)
            coef = a.coef
        avoid_ids = np.concatenate(avoid_ids_list)
        
        if self.nt:
            directions = np.array([0, 0, 1])
            projections = np.sum(leadfield * directions, axis=2)
            field_data = np.mean(projections, axis=2)
            field_data = field_data[:, :, np.newaxis] * np.ones((1, 1, 3))
        else:
            field_data = cp.array(leadfield)

        aal_regions = index
        # region_volumes = {}

        # for region in tqdm(np.unique(aal_regions)):
        #     region_volumes[region] = np.where(aal_regions == region)[0].size

        cur_max = self.max_individual_current
        cur_min = self.min_individual_current
        precision = self.precision

        currents1 = np.expand_dims(np.arange(cur_min, cur_max, precision), axis=-1)
        currents2 = cur_max * np.ones_like(currents1)
        usable_currents = np.concatenate([currents1, currents2], axis=-1)
        usable_currents = usable_currents[usable_currents[:,0] <= usable_currents[:,1],:]
        
        aal_regions_gpu = cp.array(aal_regions)
        
        ga_objective_df = lambda x, **kwargs: objective_df(x,
                                                           field_data, 
                                                           roi_ids,
                                                           avoid_ids,
                                                           coef,
                                                           aal_regions=aal_regions_gpu, 
                                                           currents=usable_currents, 
                                                           threshold=0.2,
                                                           nt=self.nt)

        var_type = np.array([['int']] * self.active_electrodes)
        bounds = np.array([[0, 74]] * self.active_electrodes)

        result = ga(function=ga_objective_df,
                    dimension=bounds.shape[0],
                    variable_type_mixed=var_type, 
                    variable_boundaries=bounds, 
                    algorithm_parameters=algorithm_param, 
                    function_timeout=time_threshold, 
                    convergence_curve=False)
        
        print(f"[optimize]INFO: run generic algorithm")
        result.run()

        # pso = PSO(func=ga_objective_df, 
        #           dim=self.active_electrodes, 
        #           pop=40, 
        #           max_iter=5, 
        #           lb=[0, 0, 0, 0], 
        #           ub=[74, 74, 74, 74],
        #           w=0.8, 
        #           c1=0.5,
        #           c2=0.5)

        # pso.run()

        # plt.plot(pso.gbest_y_hist)
        # plt.show()

        # print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

        # opt_elecs = pso.gbest_x

        convergence = result.report
        solution = result.output_dict
        opt_elecs = solution['variable'].astype(np.int32)

        best_f = float('inf')
        best_curr = usable_currents[0]

        for i in range(len(usable_currents)):
            curr_f = objective_df(opt_elecs, 
                                  field_data, 
                                  roi_ids,
                                  avoid_ids,
                                  coef,
                                  aal_regions=aal_regions_gpu, 
                                  currents=usable_currents, 
                                  threshold=0.2,
                                  nt=self.nt)
        
            
            if curr_f < best_f:
                best_curr = usable_currents[i]
                best_f = curr_f
        
        # Solution  (add 1 as Python is 0-indexed)
        electrodes, currents = (opt_elecs + 1).tolist(), best_curr.tolist()
        
        # 读取id2elec.json, 将id转换为对应的电极名称
        with open('id2elec.json', 'r') as file:
            id2elec = json.load(file)
            
        electrodes = [id2elec[str(elec)] for elec in electrodes]

        currents = [currents[0], -currents[0], currents[1], -currents[1]]
        
        sol = {'electrodes': electrodes, 'currents': currents}
        print(f"[optimize]INFO: solution: \n {sol}")

        # save df_dict to json
        with open("result.json", "w") as json_file:
            json.dump(sol, json_file, indent=4)

        electrodepair = [
            [electrodes[0], electrodes[1], currents[0]],
            [electrodes[2], electrodes[3], currents[2]]
        ]
        plot_elec(electrodepair)


        # Simple QP-style optimization
        # opt_problem = optimization_methods.TESLinearElecConstrained(
        #         self.active_electrodes, self.leadfield,
        #         max_total_current, max_individual_current, weights)

        # for t in self.target:
        #     opt_problem.add_linear_constraint(
        #         *t.get_indexes_and_directions(), t.intensity,
        #         t.get_weights()
        #     )

        logger.log(25, f"[optimization   ]INFO: end:{time.ctime()}, cost: {time.time()-startTime:.4f} seconds")
        logger.log(25, '\n' + str({'electrodes': electrodes, 'currents': currents}))

        return currents

    def field(self, currents):
        ''' Outputs the electric fields caused by the current combination

        Parameters
        -----------
        currents: N_elec x 1 ndarray
            Currents going through each electrode, in A. Usually from the optimize
            method. The sum should be approximately zero

        Returns
        ----------
        E: simnibs.mesh.NodeData or simnibs.mesh.ElementData
            NodeData or ElementData with the field caused by the currents
        '''

        assert np.isclose(np.sum(currents), 0, atol=1e-5), 'Currents should sum to zero'
        E = np.einsum('ijk,i->jk', self.leadfield, currents[1:])

        if self.lf_type == 'node':
            E = mesh_io.NodeData(E, self.field_name, mesh=self.mesh)

        if self.lf_type == 'element':
            E = mesh_io.ElementData(E, self.field_name, mesh=self.mesh)

        return E

    def electrode_geo(self, fn_out, currents=None, mesh_elec=None, elec_tags=None,
                      elec_positions=None):
        ''' Creates a mesh with the electrodes and their currents

        Parameters
        ------------
        currents: N_elec x 1 ndarray (optional)
            Electric current values per electrode. Default: do not print currents
        mesh_elec: simnibs.mesh.Msh (optional)
            Mesh with the electrodes. Default: look for a mesh called mesh_electrodes in
            self.leadfield_hdf
        elec_tags: N_elec x 1 ndarray of ints (optional)
            Tags of the electrodes corresponding to each leadfield column. The first is
            the reference electrode. Default: load at the attribute electrode_tags in the
            leadfield dataset
        elec_positions: N_elec x 3 ndarray of floats (optional)
            Positions of the electrodes in the head. If mesh_elec is not defined, will
            create small sphres at those positions instead.
            Default: load at the attribute electrode_pos in the leadfield dataset

        '''
        # First try to set the electrode visualizations using meshed electrodes
        if mesh_elec is None:
            if self.leadfield_hdf is not None:
                try:
                    mesh_elec = mesh_io.Msh.read_hdf5(self.leadfield_hdf, 'mesh_electrodes')
                except KeyError:
                    pass
            else:
                raise ValueError('Please define a mesh with the electrodes')

        if elec_tags is None and mesh_elec is not None:
            if self.leadfield_hdf is not None:
                with h5py.File(self.leadfield_hdf, 'r') as f:
                    elec_tags = f[self.leadfield_path].attrs['electrode_tags']
            else:
                raise ValueError('Please define the electrode tags')

        # If not, use point electrodes
        if mesh_elec is None and elec_positions is None:
            if self.leadfield_hdf is not None:
                with h5py.File(self.leadfield_hdf, 'r') as f:
                    elec_positions = f[self.leadfield_path].attrs['electrode_pos']
            else:
                raise ValueError('Please define the electrode positions')

        if mesh_elec is not None:
            elec_pos = self._electrode_geo_triangles(fn_out, currents, mesh_elec, elec_tags)
            # elec_pos is used for writing electrode names
        elif elec_positions is not None:
            self._electrode_geo_points(fn_out, currents, elec_positions)
            elec_pos = elec_positions
        else:
            raise ValueError('Neither mesh_elec nor elec_positions defined')
        if self.leadfield_hdf is not None:
            with h5py.File(self.leadfield_hdf, 'r') as f:
                try:
                    elec_names = f[self.leadfield_path].attrs['electrode_names']
                    elec_names = [n.decode() if isinstance(n,bytes) else n for n in elec_names]
                except KeyError:
                    elec_names = None

            if elec_names is not None:
                mesh_io.write_geo_text(
                    elec_pos, elec_names,
                    fn_out, name="electrode_names", mode='ba')

    def _electrode_geo_triangles(self, fn_out, currents, mesh_elec, elec_tags):
        if currents is None:
            currents = np.ones(len(elec_tags))

        assert len(elec_tags) == len(currents), 'Define one current per electrode'

        triangles = []
        values = []
        elec_pos = []
        bar = mesh_elec.elements_baricenters()
        norms = mesh_elec.triangle_normals()
        for t, c in zip(elec_tags, currents):
            triangles.append(mesh_elec.elm[mesh_elec.elm.tag1 == t, :3])
            values.append(c * np.ones(len(triangles[-1])))
            avg_norm = np.average(norms[mesh_elec.elm.tag1 == t], axis=0)
            pos = np.average(bar[mesh_elec.elm.tag1 == t], axis=0)
            pos += avg_norm * 4
            elec_pos.append(pos)

        triangles = np.concatenate(triangles, axis=0)
        values = np.concatenate(values, axis=0)
        elec_pos = np.vstack(elec_pos)
        mesh_io.write_geo_triangles(
            triangles - 1, mesh_elec.nodes.node_coord,
            fn_out, values, 'electrode_currents')

        return elec_pos

    def _electrode_geo_points(self, fn_out, currents, elec_positions):
        if currents is None:
            currents = np.ones(len(elec_positions))

        assert len(elec_positions) == len(currents), 'Define one current per electrode'
        mesh_io.write_geo_spheres(elec_positions, fn_out, currents, "electrode_currents")


    def field_mesh(self, currents):
        ''' Creates showing the targets and the field
        Parameters
        -------------
        currents: N_elec x 1 ndarray
            Currents going through each electrode, in A. Usually from the optimize
            method. The sum should be approximately zero

        Returns
        ---------
        results: simnibs.msh.mesh_io.Msh
            Mesh file
        '''
        target_fields = [t.as_field('target_{0}'.format(i+1)) for i, t in
                         enumerate(self.target)]
        weight_fields = [t.as_field('avoid_{0}'.format(i+1)) for i, t in
                         enumerate(self.avoid)]
        e_field = self.field(currents)
        e_magn_field = e_field.norm()
        if self.lf_type == 'node':
            normals = -self.mesh.nodes_normals()[:]
            e_normal_field = np.sum(e_field[:]*normals, axis=1)
            e_normal_field = mesh_io.NodeData(e_normal_field, 'normal' + e_field.field_name, mesh=self.mesh)
        m = copy.deepcopy(self.mesh)
        if self.lf_type == 'node':
            m.nodedata = [e_magn_field, e_field, e_normal_field] + target_fields + weight_fields
        elif self.lf_type == 'element':
            m.elmdata = [e_magn_field, e_field] + target_fields + weight_fields
        return m


    def write_currents_csv(self, currents, fn_csv, electrode_names=None):
        ''' Writes the currents and the corresponding electrode names to a CSV file

        Parameters
        ------------
        currents: N_elec x 1 ndarray
            Array with electrode currents
        fn_csv: str
            Name of CSV file to write
        electrode_names: list of strings (optional)
            Name of electrodes. Default: will read from the electrode_names attribute in
            the leadfield dataset
        '''
        if electrode_names is None:
            if self.leadfield_hdf is not None:
                with h5py.File(self.leadfield_hdf, 'r') as f:
                    electrode_names = f[self.leadfield_path].attrs['electrode_names']
                    electrode_names = [n.decode() if isinstance(n,bytes) else n for n in electrode_names]
            else:
                raise ValueError('Please define the electrode names')

        assert len(electrode_names) == len(currents)
        with open(fn_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            for n, c in zip(electrode_names, currents):
                writer.writerow([n, c])

    def run(self, cpus=1):
        ''' Interface to use with the function

        Parameters
        ---------------
        cpus: int (optional)
            Does not do anything, it is just here for the common interface with the
            simulation's run function
        '''
        if not self.name:
            if self.leadfield_hdf is not None:
                try:
                    name = re.search(r'(.+)_leadfield_', self.leadfield_hdf).group(1)
                except AttributeError:
                    name = 'optimization'
            else:
                name = 'optimization'
        else:
            name = self.name
        out_folder = os.path.dirname(name)
        os.makedirs(out_folder, exist_ok=True)

        # Set-up logger
        fh = logging.FileHandler(name + '.log', mode='w')
        formatter = logging.Formatter(
            '[ %(name)s - %(asctime)s - %(process)d ]%(levelname)s: %(message)s')
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

        fn_summary = name + '_summary.txt'
        fh_s = logging.FileHandler(fn_summary, mode='w')
        fh_s.setFormatter(logging.Formatter('%(message)s'))
        fh_s.setLevel(25)
        logger.addHandler(fh_s)

        fn_out_mesh = name + '.msh'
        fn_out_csv = name + '.csv'
        logger.info('Optimizing')
        logger.log(25, str(self))
        self.optimize(fn_out_mesh, fn_out_csv)
        logger.log(
            25,
            '\n=====================================\n'
            'tTIS finished running optimization\n'
            'Mesh file: {0}\n'
            'CSV file: {1}\n'
            'Summary file: {2}\n'
            '====================================='
            .format(fn_out_mesh, fn_out_csv, fn_summary))

        logger.removeHandler(fh)
        logger.removeHandler(fh_s)

        return fn_out_mesh

    def __str__(self):
        s = 'Optimization set-up\n'
        s += '===========================\n'
        s += 'Leadfield file: {0}\n'.format(self.leadfield_hdf)
        s += 'Max. total current: {0} (A)\n'.format(self.max_total_current)
        s += 'Max. individual current: {0} (A)\n'.format(self.max_individual_current)
        s += 'active electrodes: {0}\n'.format(self.active_electrodes)
        s += 'Name: {0}\n'.format(self.name)
        s += '----------------------\n'
        s += 'N targets: {0}\n'.format(len(self.target))
        s += '......................\n'.join(
            ['Target {0}:\n{1}'.format(i+1, str(t)) for i, t in
             enumerate(self.target)])
        s += '----------------------\n'
        s += 'N avoid: {0}\n'.format(len(self.avoid))
        s += '......................\n'.join(
            ['Avoid {0}:\n{1}'.format(i+1, str(t)) for i, t in
             enumerate(self.avoid)])
        return s


    def summary(self, currents):
        ''' Returns a string with a summary of the optimization

        Parameters
        ------------
        field: ElementData or NodeData
            Field of interest

        Returns
        ------------
        summary: str
            Summary of field
        '''
        s = 'Optimization Summary\n'
        s += '=============================\n'
        s += 'Total current: {0:.2e} (A)\n'.format(np.linalg.norm(currents, ord=1)/2)
        s += 'Maximum current: {0:.2e} (A)\n'.format(np.max(np.abs(currents)))
        s += 'Active electrodes: {0}\n'.format(int(np.linalg.norm(currents, ord=0)))
        field = self.field(currents)
        s += 'Field Summary\n'
        s += '----------------------------\n'
        s += 'Peak Value (99.9 percentile): {0:.2f} ({1})\n'.format(
            field.get_percentiles(99.9)[0], self.field_units)
        s += 'Mean field magnitude: {0:.2e} ({1})\n'.format(
            field.mean_field_norm(), self.field_units)
        if np.any(self.mesh.elm.elm_type==4):
            v_units = 'mm3'
        else:
            v_units = 'mm2'
        s += 'Focality: 50%: {0:.2e} 70%: {1:.2e} ({2})\n'.format(
            *field.get_focality(cuttofs=[50, 70], peak_percentile=99.9),
            v_units)
        for i, t in enumerate(self.target):
            s += 'Target {0}\n'.format(i + 1)
            s += '    Intensity specified:{0:.2f} achieved: {1:.2f} ({2})\n'.format(
                t.intensity, t.mean_intensity(field), self.field_units)
            if t.max_angle is not None:
                s += ('    Average angle across target: {0:.1f} '
                      '(max set to {1:.1f}) (degrees)\n'.format(
                      t.mean_angle(field), t.max_angle))
            else:
                s += '    Average angle across target: {0:.1f} (degrees)\n'.format(
                    t.mean_angle(field))

        for i, a in enumerate(self.avoid):
            s += 'Avoid {0}\n'.format(i + 1)
            s += '    Mean field magnitude in region: {0:.2e} ({1})\n'.format(
                a.mean_field_norm_in_region(field), self.field_units)

        return s

class tTIStarget:
    def __init__(self, mesh=None, tissues=None, lf_type=None, indexes=None, directions='normal', max_angle=None, 
                 intensity=0.2, ROInum=1, ROIname=None, ROIcoordMNI=None, ROIshape='sphere', ROIradius=5,
                 ROIalpha=float('inf'), otheralpha=float('inf')):
        # basic
        self.mesh = mesh
        self.tissues = tissues
        self.lf_type = lf_type
        self.indexes = indexes
        self.directions = directions
        self.max_angle = max_angle
        self.intensity = intensity

        # ROI
        self.ROInum = ROInum
        self.ROIname = ROIname
        self.ROIcoordMNI = ROIcoordMNI
        self.ROIshape = ROIshape
        self.ROIradius = ROIradius
        self.ROIalpha = ROIalpha
        self.otheralpha = otheralpha

        

    @property
    def directions(self):
        return self._directions

    @directions.setter
    def directions(self, value):
        if value == 'normal':
            pass
        elif value == 'none':
            value = None
        elif isinstance(value, str):
            raise ValueError(
                'Invalid value for directions: f{directions} '
                'valid arguments are "normal", "none" or an array'
            )
        if value is None and self.max_angle is not None:
            raise ValueError(
                "Can't constrain angle in magnitude optimizations"
            )
        self._directions = value


    def get_weights(self):
        assert self.lf_type is not None, 'Please set a lf_type'

        if self.lf_type == 'node':
            weights = self.mesh.nodes_volumes_or_areas().value
        elif self.lf_type == 'element':
            weights = self.mesh.elements_volumes_and_areas().value
        else:
            raise ValueError('Invalid lf_type: {0}, should be '
                             '"element" or "node"'.format(self.lf_type))

        return weights

    def get_indexes(self):
        ''' Calculates the mesh indexes corresponding to this target
        Returns
        ----------
        indexes: (n,) ndarray of ints
            0-based region indexes

        indexes: (n,3) ndarray of floats
            Target directions
        '''
        indexes, mapping = _find_indexes(self.mesh, self.lf_type,
                                         positions=self.positions,
                                         indexes=self.indexes,
                                         tissues=self.tissues,
                                         radius=self.ROIradius)

        return indexes-1


    def get_indexes_and_directions(self):
        ''' Calculates the mesh indexes and directions corresponding to this target
        Returns
        ----------
        indexes: (n,) ndarray of ints
            0-based region indexes

        indexes: (n,3) ndarray of floats
            Target directions
        '''
        indexes, mapping = _find_indexes(self.mesh, self.lf_type,
                                         positions=self.positions,
                                         indexes=self.indexes,
                                         tissues=self.tissues,
                                         radius=self.ROIradius)

        directions = _find_directions(self.mesh, self.lf_type,
                                      self.directions, indexes,
                                      mapping)

        return indexes-1, directions


    def as_field(self, name='target_field'):
        ''' Returns the target as an ElementData or NodeData field

        Parameters
        -----------
        name: str
            Name of the field. Default: 'target_field'
        Returns
        ---------
        target: ElementData or NodeData
            A vector field with a vector pointing in the given direction in the target
        '''
        if (self.positions is None) == (self.indexes is None): # negative XOR operation
            raise ValueError('Please set either positions or indexes')

        assert self.mesh is not None, 'Please set a mesh'

        if self.directions is None:
            nr_comp = 1
        else:
            nr_comp = 3

        if self.lf_type == 'node':
            field = np.zeros((self.mesh.nodes.nr, nr_comp))
            field_type = mesh_io.NodeData
        elif self.lf_type == 'element':
            field = np.zeros((self.mesh.elm.nr, nr_comp))
            field_type = mesh_io.ElementData
        else:
            raise ValueError("lf_type must be 'node' or 'element'."
                             " Got: {0} instead".format(self.lf_type))

        indexes, mapping = _find_indexes(self.mesh, self.lf_type,
                                         positions=self.positions,
                                         indexes=self.indexes,
                                         tissues=self.tissues,
                                         radius=self.ROIradius)

        if self.directions is None:
            field[indexes-1] = self.intensity
        else:
            directions = _find_directions(
                self.mesh, self.lf_type,
                self.directions, indexes,
                mapping
            )
            field[indexes-1] = directions * self.intensity

        return field_type(field, name, mesh=self.mesh)

    def mean_intensity(self, field):
        ''' Calculates the mean intensity of the given field in this target

        Parameters
        -----------
        field: Nx3 NodeData or ElementData
            Electric field

        Returns
        ------------
        intensity: float
            Mean intensity in this target and in the target direction
        '''
        if (self.positions is None) == (self.indexes is None): # negative XOR operation
            raise ValueError('Please set either positions or indexes')

        assert self.mesh is not None, 'Please set a mesh'
        assert field.nr_comp == 3, 'Field must have 3 components'

        indexes, mapping = _find_indexes(self.mesh, self.lf_type,
                                         positions=self.positions,
                                         indexes=self.indexes,
                                         tissues=self.tissues,
                                         radius=self.ROIradius)

        f = field[indexes]
        if self.directions is None:
            components = np.linalg.norm(f, axis=1)

        else:
            directions = _find_directions(self.mesh, self.lf_type,
                                          self.directions, indexes,
                                          mapping)

            components = np.sum(f * directions, axis=1)

        if self.lf_type == 'node':
            weights = self.mesh.nodes_volumes_or_areas()[indexes]
        elif self.lf_type == 'element':
            weights = self.mesh.elements_volumes_and_areas()[indexes]
        else:
            raise ValueError("lf_type must be 'node' or 'element'."
                             " Got: {0} instead".format(self.lf_type))

        return np.average(components, weights=weights)

    def mean_angle(self, field):
        ''' Calculates the mean angle between the field and the target

        Parameters
        -----------
        field: Nx3 NodeData or ElementData
            Electric field

        Returns
        ------------
        angle: float
            Mean angle in this target between the field and the target direction, in
            degrees
        '''
        if (self.positions is None) == (self.indexes is None): # negative XOR operation
            raise ValueError('Please set either positions or indexes')

        assert self.mesh is not None, 'Please set a mesh'
        assert field.nr_comp == 3, 'Field must have 3 components'
        if self.directions is None:
            return np.nan

        indexes, mapping = _find_indexes(self.mesh, self.lf_type,
                                         positions=self.positions,
                                         indexes=self.indexes,
                                         tissues=self.tissues,
                                         radius=self.ROIradius)

        directions = _find_directions(self.mesh, self.lf_type,
                                      self.directions, indexes,
                                      mapping)
        if self.intensity < 0:
            directions *= -1
        f = field[indexes]
        components = np.sum(f * directions, axis=1)
        norm = np.linalg.norm(f, axis=1)
        tangent = np.sqrt(norm ** 2 - components ** 2)
        angles = np.rad2deg(np.arctan2(tangent, components))
        if self.lf_type == 'node':
            weights = self.mesh.nodes_volumes_or_areas()[indexes]
        elif self.lf_type == 'element':
            weights = self.mesh.elements_volumes_and_areas()[indexes]
        else:
            raise ValueError("lf_type must be 'node' or 'element'."
                             " Got: {0} instead".format(self.lf_type))
        weights *= norm
        return np.average(angles, weights=weights)

    def __str__(self):
        s = ('positions: {0}\n'
             'indexes: {1}\n'
             'directions: {2}\n'
             'radius: {3}\n'
             'intensity: {4}\n'
             'max_angle: {5}\n'
             'tissues: {6}\n'
             .format(
                 str(self.positions),
                 str(self.indexes),
                 str(self.directions),
                 self.ROIradius,
                 self.intensity,
                 str(self.max_angle),
                str(self.tissues)))
        return s



class tTISavoid:
    ''' List of positions to be avoided by optimizer

    Attributes
    -------------
    positions: Nx3 ndarray
        List of positions to be avoided, in x, y, z coordinates and in the subject space.
        Will find the closest mesh points
    indexes: Nx1 ndarray of ints
        Indexes (1-based) of elements/nodes to be avoided. Overwrites positions
    weight : float (optional)
        Weight to give to avoid region. The larger, the more we try to avoid it. Default:
        1e3
    radius: float (optional)
        Radius of region. All the elements/nodes within the given radius of the indexes
        will be included.
    tissues: list or None (Optional)
        Tissues to be included in the region. Either a list of integer with tissue tags or None
        for all tissues. Default: None
    '''
    def __init__(self, positions=None, indexes=None,
                 weight=1e3, radius=2, tissues=None,
                 mesh=None, shape='sphere',lf_type=None):
        self.lf_type = lf_type
        self.mesh = mesh
        self.shape = shape
        self.radius = radius
        self.tissues = tissues
        self.positions = positions
        self.indexes = indexes
        self.weight = weight


    def _get_avoid_region(self):
        if (self.indexes is not None) or (self.positions is not None):
            indexes, _ = _find_indexes(self.mesh, self.lf_type,
                                       positions=self.positions,
                                       indexes=self.indexes,
                                       tissues=self.tissues,
                                       radius=self.radius)
            return indexes
        elif self.tissues is not None:
            if self.lf_type == 'element':
                return self.mesh.elm.elm_number[
                    np.isin(self.mesh.elm.tag1, self.tissues)]
            elif self.lf_type == 'node':
                return self.mesh.elm.nodes_with_tag(self.tissues)
        else:
            raise ValueError('Please define either indexes/positions or tissues')


    def avoid_field(self):
        ''' Returns a field with self.weight in the target area and
        weight=1 outside the target area

        Returns
        ------------
        w: float, >= 1
            Weight field
        '''
        assert self.mesh is not None, 'Please set a mesh'
        assert self.lf_type is not None, 'Please set a lf_type'
        assert self.weight >= 0, 'Weights must be >= 0'
        if self.lf_type == 'node':
            f = np.ones(self.mesh.nodes.nr)
        elif self.lf_type == 'element':
            f = np.ones(self.mesh.elm.nr)
        else:
            raise ValueError("lf_type must be 'node' or 'element'."
                             " Got: {0} instead".format(self.lf_type))

        indexes = self._get_avoid_region()
        f[indexes - 1] = self.weight
        if len(indexes) == 0:
            raise ValueError('Empty avoid region!')

        return f
    

    def get_indexes(self):
        ''' Calculates the mesh indexes and directions corresponding to this avoid
        Returns
        ----------
        indexes: (n,) ndarray of ints
            0-based region indexes

        indexes: (n,3) ndarray of floats
            avoid directions
        '''
        indexes, mapping = _find_indexes(self.mesh, self.lf_type,
                                         positions=self.positions,
                                         indexes=self.indexes,
                                         tissues=self.tissues,
                                         radius=self.radius)

        return indexes-1

    def as_field(self, name='weights'):
        ''' Returns a NodeData or ElementData field with the weights

        Paramets
        ---------
        name: str (optional)
            Name for the field

        Returns
        --------
        f: NodeData or ElementData
            Field with weights
        '''
        w = self.avoid_field()
        if self.lf_type == 'node':
            return mesh_io.NodeData(w, name, mesh=self.mesh)
        elif self.lf_type == 'element':
            return mesh_io.ElementData(w, name, mesh=self.mesh)


    def mean_field_norm_in_region(self, field):
        ''' Calculates the mean field magnitude in the region defined by the avoid structure

        Parameters
        -----------
        field: ElementData or NodeData
            Field for which we calculate the mean magnitude
        '''
        assert self.mesh is not None, 'Please set a mesh'
        assert self.lf_type is not None, 'Please set a lf_type'
        indexes = self._get_avoid_region()
        v = np.linalg.norm(field[indexes], axis=1)
        if self.lf_type == 'node':
            weight = self.mesh.nodes_volumes_or_areas()[indexes]
        elif self.lf_type == 'element':
            weight = self.mesh.elements_volumes_and_areas()[indexes]
        else:
            raise ValueError("lf_type must be 'node' or 'element'."
                             " Got: {0} instead".format(self.lf_type))

        return np.average(v, weights=weight)

    def __str__(self):
        s = ('positions: {0}\n'
             'indexes: {1}\n'
             'radius: {2}\n'
             'weight: {3:.1e}\n'
             'tissues: {4}\n'
             .format(
                 str(self.positions),
                 str(self.indexes),
                 self.radius,
                 self.weight,
                 str(self.tissues)))
        return s
    












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


# 线程池
class Thread_prepare_headandLF(QThread):
    # 定义一个自定义信号，用于向主线程发送消息
    thread_signal = pyqtSignal(str)
    progressBar_signal = pyqtSignal(int)
    def __init__(self):
        super().__init__()


    def run(self, cfg=None):
        self.progressBar_signal.emit(0)
        
        startTime = time.time()
        self.thread_signal.emit(f"[prepareHead]INFO: start: {time.ctime()}")
        
        # load config if not given
        if not cfg:
            config_path = 'config_tTIS.json'
            with open(config_path, 'r', encoding='utf-8') as file:
                cfg = json.load(file)
        data_path = cfg['data_path']
        subMark = cfg['subMark']
        self.thread_signal.emit(f"[prepareHead]INFO: data_path: {data_path}")
        self.thread_signal.emit(f"[prepareHead]INFO: check T1&T2 -> T1:must | T2:recommended")
        self.progressBar_signal.emit(20)

        # 1. creating head models with charm
        # Check T1
        T1_path = data_path + "\*_T1.nii*"
        T1_path = glob.glob(T1_path)[0]
        if os.path.isfile(T1_path):
            self.thread_signal.emit(f"[prepareHead]INFO: T1_file exists, path:{T1_path}")
        else:
            self.thread_signal.emit(f"[prepareHead   ]ERROR: T1_path not found:{T1_path}")
        
        # Check T2
        T2_path = data_path + "\*_T2.nii*"
        T2_path = glob.glob(T2_path)[0]
        if os.path.isfile(T2_path):
            self.thread_signal.emit(f"[prepareHead]INFO: T2_file exists, path:{T2_path}")
            # run charm with T1 and T2
            simNIBS_charm(subMark, T1_path, T2_path)
        else:
            self.thread_signal.emit(f"[prepareHead]WARNING: T2_path not found:{T2_path}")
            # run charm with T1 only
            simNIBS_charm(subMark, T1_path)
        
        # 2. prepare leadfield of gray matter middle surface
        workspace = os.path.dirname(data_path)
        simNIBS_lf_GM(subMark=subMark, workspace=workspace)
        self.progressBar_signal.emit(50)

        # 3. prepare leadfield of gray and white matter(tetrahedron)
        simNIBS_lf_GMWM_tet(subMark=subMark, workspace=workspace)
        self.progressBar_signal.emit(90)


        self.thread_signal.emit(f"[prepareHead]INFO: end:{time.ctime()}, cost: {time.time()-startTime:.4f} seconds")
        # 进度条
        self.progressBar_signal.emit(100)



class Thread_optimization_tTIS(QThread):
    # 定义一个自定义信号，用于向主线程发送消息
    thread_signal = pyqtSignal(str)
    progressBar_signal = pyqtSignal(int)


    def __init__(self):
        super().__init__()

    def run(self, cfg=None):
        self.progressBar_signal.emit(10)
        startTime = time.time()
        self.thread_signal.emit(f"[optimization]INFO: start: {time.ctime()}")

        # load config if not given
        if not cfg:
            config_path = 'config_tTIS.json'
            with open(config_path, 'r', encoding='utf-8') as file:
                cfg = json.load(file)
        data_path = cfg['data_path']
        subMark = cfg['subMark']
        output_path = cfg['output_path']
        self.progressBar_signal.emit(10)

        # check head model
        workspace = os.path.dirname(data_path)
        m2m_path = os.path.join(workspace, 'm2m_ernie')
        msh_path = os.path.join(m2m_path, f"{subMark}.msh")
        if os.path.exists(m2m_path):
            if os.path.isfile(msh_path):
                self.thread_signal.emit(f"[optimization]INFO: .msh_file already exists: {msh_path}")
            else:
                shutil.rmtree(m2m_path)
                self.thread_signal.emit(f"[optimization]ERROR: .msh_file not exist in {msh_path}")
        else:
            self.thread_signal.emit(f"[optimization]ERROR: .msh_file not exist in {msh_path}")

        opt_method = cfg['Method']
        active_electrodes = cfg['initialElecNum']
        max_individual_current = cfg['cur_max']
        min_individual_current = cfg['cur_min']
        max_total_current = cfg['cur_sum']
        precision = cfg['precision']

        type = cfg['type']

        if opt_method == 'GeneticAlgorithm':
            method_para = cfg['GA_para']

        case = cfg['case']

        # Initialize opt
        opt = tTIS_opt()
        
        if case == 1:
            ROIcfg = cfg['case1']
            opt.name = 'optimization/single_target'
        elif case == 2:
            ROIcfg = cfg['case2']
            opt.name = 'optimization/multi_target'

        ROInum = ROIcfg['num']
        ROIname = ROIcfg['name']
        ROIcoordMNI = ROIcfg['coordMNI']
        ROIshape = ROIcfg['shape']
        ROIradius = ROIcfg['radius']
        intensity = float(ROIcfg['intensity'])

        if type == 'tri':
            lf = 'leadfield'
        elif type == 'tet':
            lf = 'leadfield_tet'

        opt.leadfield_hdf = os.path.join(workspace, lf, f'{subMark}_leadfield_EEG10-10_UI_Jurak_2007.hdf5')

        # opt parameters set
        opt.max_total_current = max_total_current
        opt.max_individual_current = max_individual_current
        opt.min_individual_current = min_individual_current
        opt.active_electrodes = active_electrodes
        opt.precision = precision
        opt.opt_method = opt_method
        opt.nt = bool(cfg['nt'])
        opt.method_para = method_para

        # define optimization target(ROI)
        for i in range(ROInum):
            self.thread_signal.emit(f"[optimization]INFO: ROI {ROIname[i]} | {ROIcoordMNI[i]}")
            target = opt.add_target()
            # position of target, in subject space!
            target.positions = simnibs.mni2subject_coords(ROIcoordMNI[i], m2m_path)
            target.ROIshape = ROIshape[i]
            target.ROIradius = ROIradius[i]
            target.intensity = intensity
        self.progressBar_signal.emit(20)

        avoid_cfg = cfg['avoid']
        avoid_num = avoid_cfg['num']
        avoid_name = avoid_cfg['name']
        avoid_coordMNI = avoid_cfg['coordMNI']
        avoid_shape = avoid_cfg['shape']
        avoid_radius = avoid_cfg['radius']
        avoid_coef = avoid_cfg['coef']

        for i in range(avoid_num):
            self.thread_signal.emit(f"[optimization]INFO: add avoid {avoid_name[i]}")
            avoid = opt.add_avoid()
            avoid.positions = simnibs.mni2subject_coords(avoid_coordMNI[i], m2m_path)
            avoid.ROIshape = avoid_shape[i]
            avoid.ROIradius = avoid_radius[i]
            avoid.coef = avoid_coef
        self.progressBar_signal.emit(30)
        self.thread_signal.emit(f"It takes a long time. Pls wait...")
        opt.run()
        self.progressBar_signal.emit(90)
        
        self.thread_signal.emit(f"[optimization]INFO: end:{time.ctime()}, cost: {time.time()-startTime:.4f} seconds")
        self.progressBar_signal.emit(100)



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

        startTime = time.time()
        self.thread_signal.emit(f"[optimization]INFO: start: {time.ctime()}")

        with open('result.json', 'r') as json_file:
            result = json.load(json_file)
    
        TIpair1 = [result['electrodes'][0], result['electrodes'][1], result['currents'][0]]
        TIpair2 = [result['electrodes'][2], result['electrodes'][3], result['currents'][2]]

        config_path = 'config_tTIS.json'
        with open(config_path, 'r', encoding='utf-8') as file:
            cfg = json.load(file)

        data_path = cfg['data_path']
        subMark = cfg['subMark']
        workspace = os.path.dirname(data_path)
        m2m_path = os.path.join(workspace, 'm2m_ernie')
        output_path = cfg['output_path']
        os.makedirs(output_path, exist_ok=True)

        lf = 'leadfield_tet'
        leadfield_hdf = os.path.join(workspace, lf, f'{subMark}_leadfield_EEG10-10_UI_Jurak_2007.hdf5')
        self.progressBar_signal.emit(15)

        leadfield, mesh, idx_lf = TI.load_leadfield(leadfield_hdf=leadfield_hdf)
        mout = copy.deepcopy(mesh)
        self.progressBar_signal.emit(30)
        
        case = cfg['case']

        if case == 1:
            ROIcfg = cfg['case1']
        elif case == 2:
            ROIcfg = cfg['case2']

        ROIcoordMNI = ROIcfg['coordMNI']

        position = simnibs.mni2subject_coords(ROIcoordMNI[0], m2m_path)

        # get fields for the two pairs
        ef1 = TI.get_field_gpu(TIpair1, leadfield, idx_lf)
        ef2 = TI.get_field_gpu(TIpair2, leadfield, idx_lf)
        self.progressBar_signal.emit(45)

        hlpvar = mesh_io.ElementData(cp.asnumpy(ef1), mesh=mout)
        # mout.add_element_field(hlpvar.norm(),'E_magn1')
        hlpvar = mesh_io.ElementData(cp.asnumpy(ef2), mesh=mout)
        # mout.add_element_field(hlpvar.norm(),'E_magn2')

        TImax = TI.get_maxTI_gpu(ef1, ef2)
        # TImax = normalize_to_range(TImax, new_min=-0.2, new_max=0.2)
        hlpvar = mesh_io.ElementData(cp.asnumpy(TImax), mesh=mout)
        # mout.add_element_field(hlpvar.norm(), 'TImax_ori')
        mout.add_element_field(hlpvar.norm(), 'TImax')
        self.progressBar_signal.emit(60)

        del ef1, ef2, TImax
        cp._default_memory_pool.free_all_blocks()

        # rm_list_node = []
        # rm_list_elm = []

        # for idx, coords in enumerate(mout.elm_node_coords()):
        #     # remove elem in 2nd quadrant
        #     if (coords[:, 0] > 0).any() and (coords[:, 1] < 0).any() and (coords[:, 2] < 20).any():
        #         rm_list_elm.append(idx + 1)

        # for idx, coord in enumerate(mout.nodes.node_coord):
        #     if (coord[0] > 0) or (coord[1] < 0) or (coord[2] < 20):
        #         rm_list_node.append(idx + 1)

        # mout = mout.crop_mesh(nodes=rm_list_node, elements=rm_list_elm)
        self.progressBar_signal.emit(75)

        print(f"[performones]INFO: end:{time.ctime()}, cost: {time.time()-startTime:.4f} seconds")
        save_path = os.path.join(output_path, 'TI_via_leadfields.msh')
        mesh_io.write_msh(mout, save_path)
        self.progressBar_signal.emit(90)

        v = mout.view(visible_fields='TImax',)
        v.write_opt(save_path)

        print(f"[performones]INFO: end:{time.ctime()}, cost: {time.time()-startTime:.4f} seconds")
        self.progressBar_signal.emit(100)
        mesh_io.open_in_gmsh(save_path, True)





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
        with open('config_tTIS.json', 'w') as file:
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
