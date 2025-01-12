import numpy as np
import scipy
import os 
import h5py
import GA
import time 
import argparse 
import cupy as cp
from utils.utils import objective_df_cp

# Create the parser
parser = argparse.ArgumentParser(description='Your script description here.')

# Add arguments
parser.add_argument('--SID', type=str, help='Subject ID', default='TIME020')
parser.add_argument('--roi_n', type=int, help='ROI number', default=1)
parser.add_argument('--train', action='store_true',  help='Run in train mode')
parser.add_argument('--debug', action='store_true',  help='Run in debug mode')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
# Parse the arguments
args = parser.parse_args()

def main(SID, roi_n, headmodel_path, train=False, debug=False):

    # Set algorithm parameters
    ALG_PARAMS = {'max_num_iteration': 25,
                        'population_size': 100,
                        'mutation_probability': 0.4,
                        'elit_ratio': 0.05,
                        'crossover_probability': 0.5,
                        'parents_portion': 0.2,
                        'crossover_type': 'uniform',
                        'max_iteration_without_improv': None,
                        'opt_thresh': 0.2,
                        'cur_min': 0.05, 
                        'cur_max': 2,
                        'cur_step': 0.01,
                        'SID': SID,
                        'roi_n': roi_n
                    }
   
    
    # Keywords related to GA 
    GA_keywords = ['max_num_iteration',
                        'population_size',
                        'mutation_probability',
                        'elit_ratio',
                        'crossover_probability',
                        'parents_portion',
                        'crossover_type',
                        'max_iteration_without_improv']
                        
    # Load model
    model = scipy.io.loadmat(os.path.join(headmodel_path, f'{SID}_stats.mat'))
    # Load region indices CI
    aal_regions = np.squeeze(model['i_brain'])
    # Get number of elements
    n_elems = len(aal_regions)
    # Set brain mask 
    brain_mask = aal_regions > 0
    # Load labels
    ROIs_file = scipy.io.loadmat(os.path.join(headmodel_path, f'ROI/{SID}_ROIs.mat'))
    # labels indices
    ROIs_masks = ROIs_file['ROIs']['index'][0,0].toarray()
    # Load types
    roi_type = ROIs_file['ROIs']['type'][0,0][roi_n-1]
    # ROI mask
    roi_mask = ROIs_masks[:,roi_n-1] == 1
    # Set field type accoridng to ROI region type 
    if roi_type == 'ctx':
        print('Cortical ROI, running the comb direction')
        ALG_PARAMS["field_mode"] = 'comb'
    else:
        print('Non-cortical ROI, running the free direction')
        ALG_PARAMS["field_mode"] = 'free'
 
    # Load Volumes
    volumes = np.squeeze(scipy.io.loadmat(os.path.join(headmodel_path, f'{SID}_stats.mat'))['elem']['volume'][:][0,0])

    # load lead field matrix
    with h5py.File(os.path.join(headmodel_path,  f'leadfield/{SID}_LFM_E.mat'), 'r') as file:
        volLFM = file['LFM_E'][:][:,:n_elems,:]
    print('Loaded LFM!')
    # Get number of electrodes and elements 
    _, _, n_elecs = volLFM.shape
    # If usuing preferred direction load the vectors and compute inner-products 
    if ALG_PARAMS['field_mode'] in ['pref', 'comb']:
        # Load EV
        with h5py.File(os.path.join(headmodel_path,  f'prefdir/{SID}_cortex_ev.mat'), 'r') as file:
            volPrefVec = file['prefdir'][:]
            if volPrefVec.shape[0] == 3:
                volPrefVec = volPrefVec.T
            volPrefVec = volPrefVec[:n_elems]
        
    if ALG_PARAMS['field_mode'] == 'pref':
        # Update brain mask to only GM 
        brain_mask = aal_regions == 2
        # Initialize field 
        field_data = 0.
        # Compute LFM along the pref direction 
        for dir_i in range(3):
            # Compute LFM along this direction 
            field_data += np.multiply(volLFM[dir_i], volPrefVec[:n_elems,dir_i:dir_i+1])
        # Transpose to have [n_elecs, n_elems]
        field_data = field_data.T
        # Field data in the brain 
        brain_field_data = field_data[:,brain_mask]
        
    else:
        # Reshape to (# of elecs, # of elems, 3)
        field_data = np.transpose(volLFM, (2, 1, 0))
        # Field data in the brain 
        brain_field_data = field_data[:,brain_mask,:]
   
    # Set ROI label index
    roi_label = aal_regions.max() + 1
    # Modify regions 
    aal_regions[roi_mask] = roi_label
    # Set region volumes
    region_volumes = np.array([volumes[aal_regions == lbl].sum() for lbl in np.unique(aal_regions[brain_mask])]) 
    # File prefix
    prefix = f'{SID}_{n_elecs}elecs_roi_n{roi_n}_TI_{ALG_PARAMS["field_mode"]}_GA'

    print(f'Running for ROI {roi_n}, {SID}, and {ALG_PARAMS["field_mode"]}')
    if train:
        # Start time
        t_st = time.time()
        # Run GA
        sol, convergence_report = GA.runGA(brain_field_data, 
                np.array([roi_label]), 
                aal_regions[brain_mask], 
                region_volumes, 
                dict([(key,ALG_PARAMS[key]) for key in GA_keywords]), 
                cur_min=ALG_PARAMS['cur_min'], 
                cur_max=ALG_PARAMS['cur_max'], 
                cur_step=ALG_PARAMS['cur_step'], 
                opt_threshold=ALG_PARAMS['opt_thresh'], 
                pref_dir=volPrefVec[brain_mask] if ALG_PARAMS['field_mode'] in ['pref', 'comb'] else None,
                cortex_region=3,
                mode=ALG_PARAMS['field_mode'],
                parallel=True,
                gpu=False)
        # End time
        t_end = time.time()
        print(f'Found the optimal in {(t_end - t_st) // 60}(min)')
        # Save solution 
        scipy.io.savemat(os.path.join(data_path, f"Solutions/{prefix}.mat"), 
                sol)
        # Save log
        np.savez(os.path.join(data_path, f"Logs/{prefix}_convergence_report"), 
                            report=convergence_report,
                            time=t_end - t_st)
        


    if debug:
        
        
       # Usable currents 
        currents1 = np.expand_dims(np.arange(ALG_PARAMS["cur_min"], ALG_PARAMS["cur_max"], ALG_PARAMS["cur_step"]), axis=-1)
        currents2 = 2 * np.ones_like(currents1)
        usable_currents = np.concatenate([currents1, currents2], 
                                    axis=-1)
        usable_currents = usable_currents[usable_currents[:,0] <= usable_currents[:,1],:]
        usable_currents = usable_currents[usable_currents[:,0] <= usable_currents[:,1],:]
        print(usable_currents)
        for _ in range(10):
            x0 = np.random.randint(0, n_elecs - 1, (4,))
            print('Number of currents combinations: ', len(usable_currents))
            # t0 = time.time()
            # y0_cp = objective_df_cp(x0, 
            #                         brain_field_data, 
            #                         np.array([roi_label]), 
            #                         cp.array(aal_regions[brain_mask]), 
            #                         region_volumes, 
            #                         usable_currents, 
            #                         ALG_PARAMS['opt_thresh'], 
            #                         mode=ALG_PARAMS['field_mode'])
            # t1 = time.time()
            # print(f'cp took {t1-t0}')
            # t0 = time.time()
            # y0_np = GA.objective_df_np(x0, 
            #                         brain_field_data, 
            #                         np.array([roi_label]), 
            #                         aal_regions[brain_mask], 
            #                         region_volumes, 
            #                         usable_currents, 
            #                         ALG_PARAMS['opt_thresh'],
            #                         mode=ALG_PARAMS['field_mode'])
            # print(f'np took {t1-t0}')
            t1 = time.time()
            
            y0 = GA.objective_df(x0, 
                                    brain_field_data, 
                                    np.array([roi_label]), 
                                    aal_regions[brain_mask], 
                                    region_volumes, 
                                    usable_currents, 
                                    ALG_PARAMS['opt_thresh'], 
                                    pref_dir = volPrefVec[brain_mask] if ALG_PARAMS['field_mode'] in ['pref', 'comb'] else None,
                                    mode=ALG_PARAMS['field_mode'], 
                                    parallel=True)
            t2 = time.time()
            print(x0, y0)
            print(f'Function evaluation took {round(t2-t1,2)}(s)')

  

    print('Done!')


if __name__ == "__main__":
    # Define datapath
    data_path = f'/work/pi_sumientra_rampersad_umb_edu/Projects/TIS/Data'
    # Path for head-models
    headmodel_path = os.path.join(data_path, f'HeadModels/{args.SID}')
    # Run main func 
    main(args.SID, 2, headmodel_path, train=False, debug=True)

    print('Main module done!')

     

