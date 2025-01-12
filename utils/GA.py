"""
Optimization objective function and genetic algorithm

@Author: boyu
@Date:   2024-12-20 18:44:11
"""

import os
import gc
import sys
#import yaml
import time
import numpy as np
try:
    import cupy as cp
except ImportError:
    print('Could not load CuPy')
    
from geneticalgorithm import geneticalgorithm as ga
from .helpers import modulation_envelope    
import time
import scipy
from concurrent.futures import ThreadPoolExecutor
from .utils import objective_df_cp
from .utils import modulation_envelope_gpu
from . import TI_utils as TI

# def objective_df(x, field_data, regions_of_interest, avoid, coef, aal_regions, currents, threshold, nt=False):  
#     if np.unique(np.round(x[:4]), return_counts=True)[1].size != 4:
#         return 100 * (np.abs((np.unique(np.round(x[:4]), return_counts=True)[1].size - 4))**2) + 10000

#     penalty = 0
#     electrodes = np.round(x[:4]).astype(np.int32)  # The first 4 indices are the electrode IDs
#     aal_regions_gpu = cp.array(aal_regions)  # Move this to GPU memory
#     # regions_of_interest_gpu = cp.array(regions_of_interest)  # Move this to GPU memory
    
#     roi = cp.isin(aal_regions, cp.array(regions_of_interest))
#     if avoid.any():
#         avoid = cp.isin(aal_regions, cp.array(avoid))
    
#     max_vals = []
#     fitness_vals = []
#     avoid_penalty_l = []

#     for current in currents:
#         # Precompute the electric field
#         e_field_base = current[0] * field_data[electrodes[0]] - current[0] * field_data[electrodes[1]]
#         e_field_df = current[1] * field_data[electrodes[2]] - current[1] * field_data[electrodes[3]]

#         # GPU arrays for electric fields
#         e_field_base_gpu = cp.array(e_field_base)
#         e_field_df_gpu = cp.array(e_field_df)

#         modulation_values = modulation_envelope_gpu(e_field_base_gpu, e_field_df_gpu, nt)

#         # Get the max value for regions of interest
#         max_vals.append(float(cp.amax(modulation_values[roi])))

#         # Calculate region sums
#         unique_regions, inverse_indices = cp.unique(aal_regions_gpu, return_inverse=True)

#         # Compute the modulation values sum for each region
#         # region_sums = cp.array([cp.sum(modulation_values[aal_regions_gpu == region]) for region in unique_regions])
#         region_sums = cp.bincount(inverse_indices, weights=modulation_values)

#         # ROI sum, non-ROI sums and avoid sum
#         # is_roi = cp.isin(unique_regions, regions_of_interest_gpu)
#         is_roi = cp.isin(unique_regions, cp.array(regions_of_interest))
#         if avoid.any():
#             is_avoid = cp.isin(unique_regions, cp.array(avoid))
        
#         roi_region_sum = cp.sum(region_sums[is_roi])
#         if avoid.any():
#             avoid_region_sum = cp.sum(region_sums[is_avoid])
#             non_roi_avoid_sum = cp.sum(region_sums[~is_roi & ~is_avoid])
#         else:
#             non_roi_avoid_sum = cp.sum(region_sums[~is_roi])

#         # Fitness measure
#         region_ratio = cp.nan_to_num(roi_region_sum / non_roi_avoid_sum)
        
#         if avoid.any():
#             avoid_ratio = cp.nan_to_num(avoid_region_sum / non_roi_avoid_sum)
#             avoid_penalty = avoid_ratio * 100000 * coef
#             avoid_penalty_l.append(float(avoid_ratio))

#         fitness_measure = region_ratio * 100000
#         fitness_vals.append(float(fitness_measure))

#     # Convert lists to arrays
#     max_vals = cp.array(max_vals)
#     fitness_vals = cp.array(fitness_vals)
#     if avoid.any():
#         avoid_penalty_l = cp.array(avoid_penalty_l)

#     # Evaluate fitness based on threshold
#     max_val_curr = cp.where(max_vals >= threshold)[0]
#     return_fitness = 0
#     if max_val_curr.size == 0:
#         penalty += 100 * ((threshold - cp.mean(max_vals))**2) + 1000
#         return_fitness = cp.amin(fitness_vals)
#     else:
#         fitness_candidate = cp.amax(fitness_vals[max_val_curr])
#         return_fitness = fitness_candidate
#     if avoid.any():
#         avoid_penalty_max = cp.amax(avoid_penalty_l)
#         res = -float(cp.round(return_fitness - penalty - avoid_penalty_max, 2))

#     else:
#         res = -float(cp.round(return_fitness - penalty, 2))
        
#     return res



def objective_df(x, field_data, regions_of_interest, avoid, coef, aal_regions, currents, threshold, nt=False):  
    # startTime = time.time()
    if np.unique(np.round(x[:4]), return_counts=True)[1].size != 4:
        return 100 * (np.abs((np.unique(np.round(x[:4]), return_counts=True)[1].size - 4))**2) + 10000

    penalty = 0
    electrodes = cp.round(x[:4]).astype(np.int32)  # The first 4 indices are the electrode IDs
    
    roi = cp.isin(aal_regions, cp.array(regions_of_interest))
    if avoid.any():
        avoid = cp.isin(aal_regions, cp.array(avoid))
    
    max_vals = []
    fitness_vals = []
    avoid_penalty_l = []

    for current in currents:
        E1_gpu = current[0] * (field_data[electrodes[0]] - field_data[electrodes[1]])
        E2_gpu = current[1] * (field_data[electrodes[2]] - field_data[electrodes[3]])
        
        if not nt:
            TImax = TI.get_maxTI_gpu(E1_gpu, E2_gpu)
        else:
            TImax = TI.get_dirTI_gpu(E1_gpu, E2_gpu, [0,0,1])

        # Get the max value for regions of interest
        max_vals.append(float(cp.amax(TImax[roi])))

        # ROI sum, non-ROI sums and avoid sum
        roi_sum = cp.sum(TImax[roi])
        if avoid.any():
            avoid_sum = cp.sum(TImax[avoid])
            non_sum = cp.sum(TImax) - roi_sum - avoid_sum
        else:
            non_sum = cp.sum(TImax) - roi_sum

        # Fitness measure
        region_ratio = cp.nan_to_num(roi_sum / non_sum)
        
        if avoid.any():
            avoid_ratio = cp.nan_to_num(avoid_sum / non_sum)
            avoid_penalty = avoid_ratio * 100000 * coef
            avoid_penalty_l.append(float(avoid_ratio))

        fitness_measure = region_ratio * 100000
        fitness_vals.append(float(fitness_measure))

        del E1_gpu, E2_gpu, TImax
        gc.collect()

    # Convert lists to arrays
    max_vals = cp.array(max_vals)
    fitness_vals = cp.array(fitness_vals)
    if avoid.any():
        avoid_penalty_l = cp.array(avoid_penalty_l)

    # Evaluate fitness based on threshold
    max_val_curr = cp.where(max_vals >= threshold)[0]
    return_fitness = 0
    if max_val_curr.size == 0:
        penalty += 100 * ((threshold - cp.mean(max_vals))**2) + 1000
        return_fitness = cp.amin(fitness_vals)
    else:
        fitness_candidate = cp.amax(fitness_vals[max_val_curr])
        return_fitness = fitness_candidate
    if avoid.any():
        avoid_penalty_max = cp.amax(avoid_penalty_l)
        res = -float(cp.round(return_fitness - penalty - avoid_penalty_max, 2))
    else:
        res = -float(cp.round(return_fitness - penalty, 2))
    
    # print(return_fitness)
    # print(f"[objective_df]INFO: end:{time.ctime()}, cost: {time.time()-startTime:.4f} seconds")
    
    del max_vals, fitness_vals, avoid_penalty_l
    gc.collect()
    
    return res
   
if __name__ == "__main__":
    OPTIMIZATION_THRESHOLD = 0.2
    # Define datapath    
    # data_path = '/Users/arminmoharrer/Library/CloudStorage/GoogleDrive-armin.moharrer@umb.edu/My Drive/TIS_Code/Data'
    # npz_arrays = np.load(os.path.join(data_path, 'sphere_19/data.npz'), 
    #                      allow_pickle=True)

    # field_data = npz_arrays['e_field']
    # prefVec = scipy.io.loadmat(os.path.join(data_path, 'Sphere_19/prefdir.mat'))['prefDir']
    # n_elecs = field_data.shape[0]
    # print(f"There are {n_elecs} electrodes.")
    # # Set GA parameters 
    # algorithm_param = {'max_num_iteration': 25,
    #                    'population_size': 100,
    #                    'mutation_probability': 0.4,
    #                    'elit_ratio': 0.05,
    #                    'crossover_probability': 0.5,
    #                    'parents_portion': 0.2,
    #                    'crossover_type': 'uniform',
    #                    'max_iteration_without_improv': None
    #                 }
    
    
    # # Load ROI
    # for roi_file in os.listdir(os.path.join(data_path, 'sphere_19/ROI')):
    #     if '0_0_0' not in roi_file:
    #         continue 
    #     # Load ROI mask
    #     aal_regions = np.squeeze(scipy.io.loadmat(os.path.join(data_path, f'sphere_19/ROI/{roi_file}'))['roi_mask']) 
    #     # Compute region volumes
    #     region_volumes = np.array([sum(aal_regions == lbl) for lbl in np.unique(aal_regions)])
    #     # Set ROI label 
    #     roi_labels = np.array([1])
    #     # File prefix
    #     prefix = f'sol_GA{'_'.join([key + '_' + str(algorithm_param[key]) for key in algorithm_param])}_opthrsh_{OPTIMIZATION_THRESHOLD}_{roi_file[:-4]}'
    #     # Set currents
    #     cur_potential_values = np.arange(.5, 4, .5)

    #     # Create meshgrid
    #     cur_x, cur_y = np.meshgrid(cur_potential_values, cur_potential_values)
    #     # Create combos 
    #     cur_all_combinations = np.hstack((cur_x.reshape((-1, 1)), cur_y.reshape((-1, 1))))
    #     # Usable currents 
    #     usable_currents = cur_all_combinations[np.where(np.sum(np.round(cur_all_combinations, 2), axis=1) == 4)[0]]

    #     ga_objective_df = lambda x, **kwargs: objective_df(x, 
    #                                                 field_data, 
    #                                                 roi_labels, 
    #                                                 aal_regions, 
    #                                                 region_volumes, 
    #                                                 usable_currents, 
    #                                                 .2,
    #                                                 parallel=False)
    #     ga_objective_np = lambda x, **kwargs: objective_df_np(x, 
    #                                                 field_data, 
    #                                                 roi_labels, 
    #                                                 aal_regions, 
    #                                                 region_volumes, 
    #                                                 usable_currents, 
    #                                                 .2,
    #                                                )
        
    #     y = ga_objective_np([1,2,3,4])
        
    #     print(f'Done for {roi_file}')
    # print("Happy Optimizing!")
