import os
import gc
import h5py
import scipy
import json
import yaml
import numpy as np
import cupy as cp
#import pandas as pd
from simnibs.utils import TI_utils as TI


def load_cfg(cfg_path: str):
    if cfg_path.endswith('.json'):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            return cfg
    elif cfg_path.endswith('.yaml'):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            return cfg
    else:
        raise Exception('[load_cfg]ERROR: Only support *.json or *.yml file')


class electrode_Pos:
    def __init__(self):
        self.id2pos = {"1": [-5.0, 0.0], "2": [-4.0, 0.0], "3": [-3.0, 0.0], "4": [-2.0, 0.0], "5": [-1.0, 0.0], "6": [0.0, 0.0], "7": [1.0, 0.0], "8": [2.0, 0.0], "9": [3.0, 0.0], "10": [4.0, 0.0], "11": [5.0, 0.0], "12": [0.0, -5.0], "13": [0.0, -4.0], "14": [0.0, -3.0], "15": [0.0, -2.0], "16": [0.0, -1.0], "17": [0.0, 1.0], "18": [0.0, 2.0], "19": [0.0, 3.0], "20": [0.0, 4.0], "21": [3.8042, 1.2361], "22": [3.2361, 2.3511], "23": [2.3511, 3.2361], "24": [1.2361, 3.8042], "25": [-1.2361, 3.8042], "26": [-2.3511, 3.2361], "27": [-3.2361, 2.3511], "28": [-3.8042, 1.2361], "29": [-3.8042, -1.2361], "30": [-3.2361, -2.3511], "31": [-2.3511, -3.2361], "32": [-1.2361, -3.8042], "33": [1.2361, -3.8042], "34": [2.3511, -3.2361], "35": [3.2361, -2.3511], "36": [3.8042, -1.2361], "37": [4.7553, 1.5451], "38": [4.0451, 2.9389], "39": [-4.0451, 2.9389], "40": [-4.7553, 1.5451], "41": [-4.7553, -1.5451], "42": [-4.0451, -2.9389], "43": [-2.9389, -4.0451], "44": [-1.5451, -4.7553], "45": [1.5451, -4.7553], "46": [2.9389, -4.0451], "47": [4.0451, -2.9389], "48": [4.7553, -1.5451], "49": [-1.1756, 3.118], "50": [-1.618, 2.1756], "51": [-1.9021, 1.118], "52": [-1.9021, -1.118], "53": [-1.618, -2.1756], "54": [-1.1756, -3.118], "55": [1.1756, 3.118], "56": [1.618, 2.1756], "57": [1.9021, 1.118], "58": [1.9021, -1.118], "59": [1.618, -2.1756], "60": [1.1756, -3.118], "61": [-2.4271, 2.2634], "62": [-2.8532, 1.1771], "63": [-2.8532, -1.1771], "64": [-2.4271, -2.2634], "65": [-0.809, 2.0878], "66": [-0.9511, 1.059], "67": [-0.9511, -1.059], "68": [-0.809, -2.0878], "69": [0.809, 2.0878], "70": [0.9511, 1.059], "71": [0.9511, -1.059], "72": [0.809, -2.0878], "73": [2.4271, 2.2634], "74": [2.8532, 1.1771], "75": [2.8532, -1.1771], "76": [2.4271, -2.2634]}
        self.elec2pos = {"T9": [-5.0, 0.0], 
                         "T7": [-4.0, 0.0], 
                         "C5": [-3.0, 0.0], 
                         "C3": [-2.0, 0.0], 
                         "C1": [-1.0, 0.0], 
                         "Cz": [0.0, 0.0], 
                         "C2": [1.0, 0.0], 
                         "C4": [2.0, 0.0], 
                         "C6": [3.0, 0.0], 
                         "T8": [4.0, 0.0], 
                         "T10": [5.0, 0.0], 
                         "Iz": [0.0, -5.0], 
                         "Oz": [0.0, -4.0], 
                         "POz": [0.0, -3.0], 
                         "Pz": [0.0, -2.0], 
                         "CPz": [0.0, -1.0], 
                         "FCz": [0.0, 1.0], 
                         "Fz": [0.0, 2.0], 
                         "AFz": [0.0, 3.0], 
                         "Fpz": [0.0, 4.0], 
                         "FT8": [3.8042, 1.2361], 
                         "F8": [3.2361, 2.3511], 
                         "AF8": [2.3511, 3.2361], 
                         "Fp2": [1.2361, 3.8042], 
                         "Fp1": [-1.2361, 3.8042], 
                         "AF7": [-2.3511, 3.2361], 
                         "F7": [-3.2361, 2.3511], 
                         "FT7": [-3.8042, 1.2361], 
                         "TP7": [-3.8042, -1.2361], "P7": [-3.2361, -2.3511], "PO7": [-2.3511, -3.2361], "O1": [-1.2361, -3.8042], "O2": [1.2361, -3.8042], "PO8": [2.3511, -3.2361], "P8": [3.2361, -2.3511], "TP8": [3.8042, -1.2361], "FT10": [4.7553, 1.5451], "F10": [4.0451, 2.9389], "F9": [-4.0451, 2.9389], "FT9": [-4.7553, 1.5451], "TP9": [-4.7553, -1.5451], "P9": [-4.0451, -2.9389], "PO9": [-2.9389, -4.0451], "I1": [-1.5451, -4.7553], "I2": [1.5451, -4.7553], "PO10": [2.9389, -4.0451], "P10": [4.0451, -2.9389], "TP10": [4.7553, -1.5451], "AF3": [-1.1756, 3.118], "F3": [-1.618, 2.1756], "FC3": [-1.9021, 1.118], "CP3": [-1.9021, -1.118], "P3": [-1.618, -2.1756], "PO3": [-1.1756, -3.118], "AF4": [1.1756, 3.118], "F4": [1.618, 2.1756], "FC4": [1.9021, 1.118], "CP4": [1.9021, -1.118], "P4": [1.618, -2.1756], "PO4": [1.1756, -3.118], "F5": [-2.4271, 2.2634], "FC5": [-2.8532, 1.1771], "CP5": [-2.8532, -1.1771], "P5": [-2.4271, -2.2634], "F1": [-0.809, 2.0878], "FC1": [-0.9511, 1.059], "CP1": [-0.9511, -1.059], "P1": [-0.809, -2.0878], "F2": [0.809, 2.0878], "FC2": [0.9511, 1.059], "CP2": [0.9511, -1.059], "P2": [0.809, -2.0878], "F6": [2.4271, 2.2634], "FC6": [2.8532, 1.1771], "CP6": [2.8532, -1.1771], "P6": [2.4271, -2.2634]}
        self.id2elec = {1: 'T9',
                        2: 'T7',
                        3: 'C5',
                        4: 'C3',
                        5: 'C1',
                        6: 'Cz',
                        7: 'C2',
                        8: 'C4',
                        9: 'C6',
                        10: 'T8',
                        11: 'T10',
                        12: 'Iz',
                        13: 'Oz',
                        14: 'POz',
                        15: 'Pz',
                        16: 'CPz',
                        17: 'FCz',
                        18: 'Fz',
                        19: 'AFz',
                        20: 'Fpz',
                        21: 'FT8',
                        22: 'F8',
                        23: 'AF8',
                        24: 'Fp2',
                        25: 'Fp1',
                        26: 'AF7',
                        27: 'F7',
                        28: 'FT7',
                        29: 'TP7',
                        30: 'P7',
                        31: 'PO7',
                        32: 'O1',
                        33: 'O2',
                        34: 'PO8',
                        35: 'P8',
                        36: 'TP8',
                        37: 'FT10',
                        38: 'F10',
                        39: 'F9',
                        40: 'FT9',
                        41: 'TP9',
                        42: 'P9',
                        43: 'PO9',
                        44: 'I1',
                        45: 'I2',
                        46: 'PO10',
                        47: 'P10',
                        48: 'TP10',
                        49: 'AF3',
                        50: 'F3',
                        51: 'FC3',
                        52: 'CP3',
                        53: 'P3',
                        54: 'PO3',
                        55: 'AF4',
                        56: 'F4',
                        57: 'FC4',
                        58: 'CP4',
                        59: 'P4',
                        60: 'PO4',
                        61: 'F5',
                        62: 'FC5',
                        63: 'CP5',
                        64: 'P5',
                        65: 'F1',
                        66: 'FC1',
                        67: 'CP1',
                        68: 'P1',
                        69: 'F2',
                        70: 'FC2',
                        71: 'CP2',
                        72: 'P2',
                        73: 'F6',
                        74: 'FC6',
                        75: 'CP6',
                        76: 'P6'}
        self.elec2id = {"T9": 1,
                        "T7": 2,
                        "C5": 3,
                        "C3": 4,
                        "C1": 5,
                        "Cz": 6,
                        "C2": 7,
                        "C4": 8,
                        "C6": 9,
                        "T8": 10,
                        "T10": 11,
                        "Iz": 12,
                        "Oz": 13,
                        "POz": 14,
                        "Pz": 15,
                        "CPz": 16,
                        "FCz": 17,
                        "Fz": 18,
                        "AFz": 19,
                        "Fpz": 20,
                        "FT8": 21,
                        "F8": 22,
                        "AF8": 23,
                        "Fp2": 24,
                        "Fp1": 25,
                        "AF7": 26,
                        "F7": 27,
                        "FT7": 28,
                        "TP7": 29,
                        "P7": 30,
                        "PO7": 31,
                        "O1": 32,
                        "O2": 33,
                        "PO8": 34,
                        "P8": 35,
                        "TP8": 36,
                        "FT10": 37,
                        "F10": 38,
                        "F9": 39,
                        "FT9": 40,
                        "TP9": 41,
                        "P9": 42,
                        "PO9": 43,
                        "I1": 44,
                        "I2": 45,
                        "PO10": 46,
                        "P10": 47,
                        "TP10": 48,
                        "AF3": 49,
                        "F3": 50,
                        "FC3": 51,
                        "CP3": 52,
                        "P3": 53,
                        "PO3": 54,
                        "AF4": 55,
                        "F4": 56,
                        "FC4": 57,
                        "CP4": 58,
                        "P4": 59,
                        "PO4": 60,
                        "F5": 61,
                        "FC5": 62,
                        "CP5": 63,
                        "P5": 64,
                        "F1": 65,
                        "FC1": 66,
                        "CP1": 67,
                        "P1": 68,
                        "F2": 69,
                        "FC2": 70,
                        "CP2": 71,
                        "P2": 72,
                        "F6": 73,
                        "FC6": 74,
                        "CP6": 75,
                        "P6": 76}

def modulation_envelope_gpu(e_field_1, e_field_2, nt=False):
    if not nt:
        envelope = cp.zeros(e_field_1.shape[0])
        
        # Calculate the angles between the two fields for each vector
        dot_angle = cp.einsum('ij,ij->i', e_field_1, e_field_2)
        cross_angle = cp.linalg.norm(cp.cross(e_field_1, e_field_2), axis=1)
        angles = cp.arctan2(cross_angle, dot_angle)
        
        # Flip the direction of the electric field if the angle between the two is greater or equal to 90 degrees
        e_field_2 = cp.where(cp.broadcast_to(angles >= cp.pi/2., (3, e_field_2.shape[0])).T, -e_field_2, e_field_2)
        
        # Recalculate the angles
        dot_angle = cp.einsum('ij,ij->i', e_field_1, e_field_2)
        cross_angle = cp.linalg.norm(cp.cross(e_field_1, e_field_2), axis=1)
        angles = cp.arctan2(cross_angle, dot_angle)
        E_minus = cp.subtract(e_field_1, e_field_2) # Create the difference of the E fields
        
        # Condition to have two times the E2 field amplitude
        max_condition_1 = cp.linalg.norm(e_field_2, axis=1) < cp.linalg.norm(e_field_1, axis=1)*cp.cos(angles)
        e1_gr_e2 = cp.where(cp.linalg.norm(e_field_1, axis=1) > cp.linalg.norm(e_field_2, axis=1), max_condition_1, False)
        
        # Condition to have two times the E1 field amplitude
        max_condition_2 = cp.linalg.norm(e_field_1, axis=1) < cp.linalg.norm(e_field_2, axis=1)*cp.cos(angles)
        e2_gr_e1 = cp.where(cp.linalg.norm(e_field_2, axis=1) > cp.linalg.norm(e_field_1, axis=1), max_condition_2, False)
        
        # Double magnitudes
        envelope = cp.where(e1_gr_e2, 2.0*cp.linalg.norm(e_field_2, axis=1), envelope) # 2E2 (First case)
        envelope = cp.where(e2_gr_e1, 2.0*cp.linalg.norm(e_field_1, axis=1), envelope) # 2E1 (Second case)
        
        # Calculate the complement area to the previous calculation
        e1_gr_e2 = cp.where(cp.linalg.norm(e_field_1, axis=1) > cp.linalg.norm(e_field_2, axis=1), cp.logical_not(max_condition_1), False)
        e2_gr_e1 = cp.where(cp.linalg.norm(e_field_2, axis=1) > cp.linalg.norm(e_field_1, axis=1), cp.logical_not(max_condition_2), False)
        
        # Cross product
        envelope = cp.where(e1_gr_e2, 2.0*(cp.linalg.norm(cp.cross(e_field_2, E_minus), axis=1)/cp.linalg.norm(E_minus, axis=1)), envelope) # (First case)
        envelope = cp.where(e2_gr_e1, 2.0*(cp.linalg.norm(cp.cross(e_field_1, -E_minus), axis=1)/cp.linalg.norm(-E_minus, axis=1)), envelope) # (Second case)
    else:
        # Calculate the values of the E field modulation envelope along the desired n direction
        E_plus = cp.add(e_field_1, e_field_2) # Create the sum of the E fields
        E_minus = cp.subtract(e_field_1, e_field_2) # Create the difference of the E fields
        envelope = cp.abs(cp.abs(E_plus) - cp.abs(E_minus))
        #envelope = cp.abs(cp.abs(cp.dot(E_plus, dir_vector)) - cp.abs(cp.dot(E_minus, dir_vector)))
    
    return cp.nan_to_num(envelope)


def objective_df_cp(x, field_data, regions_of_interest, aal_regions, region_volumes, currents, threshold, mode='free'):  
    if np.unique(np.round(x[:4]), return_counts=True)[1].size != 4:
        return 100*(np.abs((np.unique(np.round(x[:4]), return_counts=True)[1].size - 4))**2) + 10000
    
    penalty = 0
    electrodes = np.round(x[:4]).astype(np.int32) # The first 4 indices are the electrode IDs
    roi = cp.isin(aal_regions, cp.array(regions_of_interest))
    max_vals = []
    fitness_vals = []
    
    for current in currents:
        e_field_base = current[0]*field_data[electrodes[0]] - current[0]*field_data[electrodes[1]]
        e_field_df = current[1]*field_data[electrodes[2]] - current[1]*field_data[electrodes[3]]

        e_field_base_gpu = cp.array(e_field_base)
        e_field_df_gpu = cp.array(e_field_df)
        modulation_values = modulation_envelope_gpu(e_field_base_gpu, e_field_df_gpu, mode=mode)
        max_vals.append(float(cp.amax(modulation_values[roi])))
    
        roi_region_sum = 0
        non_roi_region_sum = 0

        for reg_i, region in enumerate(cp.unique(aal_regions)):
            roi = cp.where(aal_regions == region)[0]
        
            if int(region) in regions_of_interest:
                roi_region_sum += cp.sum(modulation_values[roi])/region_volumes[reg_i]
            else:
                non_roi_region_sum += cp.sum(modulation_values[roi])/region_volumes[reg_i]
        
        region_ratio = cp.nan_to_num(roi_region_sum/non_roi_region_sum)
        fitness_measure = region_ratio*10000
        fitness_vals.append(float(fitness_measure))
    
    max_vals = np.array(max_vals)
    max_val_curr = np.where(max_vals >= threshold)[0]
    fitness_vals = np.array(fitness_vals)

    return_fitness = 0
    if max_val_curr.size == 0:
        penalty += 100*((threshold - np.mean(max_vals))**2) + 1000
        return_fitness = np.amin(fitness_vals)
    else:
        fitness_candidate = np.amax(fitness_vals[max_val_curr])
        return_fitness = fitness_candidate
    
    return -float(np.round(return_fitness - penalty, 2))

def readMatFile(fpath, kword):
    try:
        with h5py.File(fpath) as file:
            out = file[kword][:]
        return out
    except:
        out = scipy.io.loadmat(fpath)[kword]
        return out

if __name__ == "__main__":
    npz_dir = 'directory of the NPZ model files'
    csv_dir = 'directory of the optimized electrode pairs'
    files = next(os.walk(csv_dir))[2]
    
    for model in files:
        model = model.split('.')[0].split('_')[-1]
        
        npz_arrays = np.load(os.path.join(npz_dir, model + '_fields_brain_reg.npz'), allow_pickle=True)
        electrode_vals = pd.read_csv(os.path.join(csv_dir, 'optimized_electrodes_' + model + '.csv'))
        
        field_data = npz_arrays['e_field']
        aal_regions = npz_arrays['aal_ids']
        region_volumes = {}
        for region in np.unique(aal_regions):
            region_volumes[region] = np.where(aal_regions == region)[0].size
        roi_ids = np.array([42])
        
        ideal_case = None
        
        cur_potential_values = np.arange(0.5, 1.55, 0.05)
        cur_x, cur_y = np.meshgrid(cur_potential_values, cur_potential_values)
        cur_all_combinations = np.hstack((cur_x.reshape((-1, 1)), cur_y.reshape((-1, 1))))
        usable_currents = cur_all_combinations[np.where(np.sum(np.round(cur_all_combinations, 2), axis=1) == 2)[0]]
        
        print(model)
        electrodes_model = electrode_vals['electrodes'].to_numpy().astype(int)
        aal_regions_gpu = cp.array(aal_regions)
        value, cur_id = objective_df_cp(electrodes_model, field_data, roi_ids, aal_regions=aal_regions_gpu, region_volumes=region_volumes, currents=usable_currents, ideal_case=ideal_case)
        
        electrode_vals['currents'] = np.round([usable_currents[cur_id][0], usable_currents[cur_id][0], usable_currents[cur_id][1], usable_currents[cur_id][1]], 2)
        electrode_vals.to_csv(os.path.join(csv_dir, 'optimized_electrodes_' + model + '.csv'))
        print('Value: {}, Current: {}, {}'.format(value, usable_currents[cur_id][0], usable_currents[cur_id][1]))
        
        del field_data
        del npz_arrays
        gc.collect()
