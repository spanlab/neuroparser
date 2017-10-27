
import os, sys, glob
from nipy.io.api import load_image

from directorytools import subjects as get_subjects
from directorytools import subject_dirs as get_subject_dirs
from graphnet_hook import GraphnetInterface, Gridsearch
from datamanager import BrainData


if __name__ == "__main__":
    
    '''
    Examples of using the graphnet interface with nifti data.
    test_data should have the following structure:
    test_data/
    |-- jk160415
    |   |-- buy_vs_not_onset.1D
    |   |-- stock_mbnf.nii
    |-- jr160430
    |   |-- buy_vs_not_onset.1D
    |   |-- stock_mbnf.nii
    |-- jr160501
    |   |-- buy_vs_not_onset.1D
    |   |-- stock_mbnf.nii
    |-- tt29.nii
    '''

    #-----------------------------------------------------------#
    # Load in subject data:
    #-----------------------------------------------------------#

    thisdir = os.getcwd()
    subject_top_dir = '/Users/span/neuroparser/test_data'

    subject_folder_names = ['jk160415', 'jr160430', 'jr160501']

    subject_directories = get_subject_dirs(topdir=subject_top_dir, prefixes=subject_folder_names)


    mask = os.path.join(subject_top_dir, 'tt29.nii')

    functional_name = 'stock_mbnf.nii'
    trial_demarcation_vector_name = 'buy_vs_not_onset.1D'

    lag = 2
    selected_trial_trs = [1,2,3,4]

    datamanager = BrainData()

    datamanager.make_masks(mask, len(selected_trial_trs))

    datamanager.create_design(subject_directories,
        functional_name,
        trial_demarcation_vector_name,
        selected_trial_trs,
        lag=lag)

    datamanager.create_XY_matrices(with_replacement=True,
        replacement_ceiling=36,
        Ybinary=[1.,-1])

    datamanager.delete_subject_design()

    graphnet = GraphnetInterface(data_obj=datamanager)

    #-----------------------------------------------------------#
    # Basic usage and dumping coefficients:
    #-----------------------------------------------------------#

    # Train a robust graphnet svm. 
    # To see how to choose which problem to solve, see
    # graphnet_hook.py:293

    # for parameter estimates frmo other data, see the Neuroimage paper, Table 1, 
    # though a grid search will almost certainly be necessary. 


    graphnet.train_graphnet(datamanager.X, datamanager.Y, 
        trial_mask=datamanager.trial_mask,
        l1=43., l2=100., l3=100., delta=0.3, adaptive=True)

    coefs = graphnet.coefficients[0].copy()

    unmasked_coefs = datamanager.unmask_Xcoefs(coefs, len(selected_trial_trs),
        slice_off_back=datamanager.X.shape[0])

    datamanager.save_unmasked_coefs(unmasked_coefs, 'graphnet_coef_map')


    #-----------------------------------------------------------#
    # Crossvalidation:
    #-----------------------------------------------------------#


    train_keyword_args = {'trial_mask':datamanager.trial_mask,
                          'l1':43., 'l2':100., 'l3':100., 'delta':0.3,
                          'adaptive':True}

    graphnet.setup_crossvalidation(folds=len(subject_folder_names), leave_mod_in=True)

    accuracies, average_accuracies, non_zero_coefs = graphnet.crossvalidate(train_keyword_args)
    print accuracies, average_accuracies


    #-----------------------------------------------------------#
    # Gridsearching:
    #-----------------------------------------------------------#

    gridsearch = Gridsearch()
    gridsearch.folds = len(subject_folder_names)
    gridsearch.initial_l1_min = 10.
    gridsearch.initial_l1_max = 60.
    gridsearch.l1_stepsizes = [5.,3.,1.]
    gridsearch.deltas = [.3,.5,.7]

    gridsearch.zoom_gridsearch(graphnet,
        name='graphnet_gridsearch',
        adaptive=True)










