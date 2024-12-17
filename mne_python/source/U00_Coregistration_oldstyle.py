# -*- coding: utf-8 -*-
"""
===============================================
S00. Coregistration and preparing trans file

This script coregisteres the MEG file with MRI
and generates the trans file (which is necessary
                              for BIDS conversion)

    Step 1: Reconstructing MRI using FreeSurfer
    Step 2: Reconstructing the scalp surface
    Step 3: Getting Boundary Element Model (BEM)
    Step 4: Getting BEM solution
    Step 5: Coregistration (Automatic prefered for camcan)

trans file is created here for later use in bids and then
the source-base analysis.
1) save the trans file in the MRI folder
2) rename and move the transfile to bids structure using
    01_bids_conversion... script

    Run recon_all on freesurfer before this script (using recon_all_fs_array.sh).
   
written by Tara Ghafari
adapted from Oscar Ferrante
==============================================
"""

import numpy as np
import os.path as op
import os
import pandas as pd

import mne
import matplotlib.pyplot as plt

platform = 'mac'  # are you running on bluebear or windows or mac?
coreg = 'auto'  # auto or manual coregistration?

# Define where to read and write the data
if platform == 'bluebear':
    rds_dir = '/rds/projects/q/quinna-camcan'
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    rds_dir = '/Volumes/quinna-camcan'
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'

epoched_dir = op.join(rds_dir, 'derivatives/meg/sensor/epoched-7min50')
info_dir = op.join(rds_dir, 'dataman/data_information')
good_sub_sheet = op.join(info_dir, 'demographics_goodPreproc_subjects.csv')

# Read only data from subjects with good preprocessed data
good_subject_pd = pd.read_csv(good_sub_sheet)
good_subject_pd = good_subject_pd.set_index('Unnamed: 0')  # set subject id codes as the index

# subject info 
meg_extension = '.fif'
meg_suffix = 'meg'
trans_suffix = 'coreg-trans'
bem_suffix = 'bem-sol' 

### AUTOMATIC COREGISTRATION ##
if coreg == 'auto':
    for i, subjectID in enumerate(good_subject_pd.index):

        fs_sub = f'sub-CC{subjectID}_T1w'  # name of fs folder for each subject

        # Specify specific file names
        fs_sub_dir = op.join(rds_dir, f'cc700/mri/pipeline/release004/BIDS_20190411/anat')  # FreeSurfer directory (after running recon all)
        deriv_folder = op.join(rds_dir, 'derivatives/meg/source/freesurfer', fs_sub[:-4])

        if not os.path.exists(deriv_folder):
            os.makedirs(deriv_folder)
        trans_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + trans_suffix + meg_extension)
        bem_fname = trans_fname.replace(trans_suffix, bem_suffix)  
        bem_figname = bem_fname.replace(meg_extension, '.png')
        coreg_figname = bem_figname.replace(bem_suffix, 'final_coreg')


        plot_kwargs = dict(subject=fs_sub, 
                        subjects_dir=fs_sub_dir,
                        surfaces="head-dense", 
                        dig=True,
                        eeg=[], 
                        meg='sensors', 
                        show_axes=True,
                        coord_frame='meg',
                        show=False)
        view_kwargs = dict(azimuth=45, 
                        elevation=90, 
                        distance=.6,
                        focalpoint=(0.,0.,0.,),
                        show=False)
        try:
            print(f'Reading subject # {i}')

            # Get Boundary Element model (BEM) solution
            """ run this section after the watershed_bem surfaces are read in freesurfer,
            (using my_recon.sh batch script)"""
            print("Creating BEM model")
            conductivity = (.3,)  # for single layer
            model = mne.make_bem_model(subject=fs_sub, 
                                    subjects_dir=fs_sub_dir,
                                    ico=4, 
                                    conductivity=conductivity)

            # BEM solution is derived from the BEM model
            bem = mne.make_bem_solution(model)
            mne.write_bem_solution(bem_fname, 
                                bem, 
                                overwrite=True, 
                                verbose=True)
            # Visualize the BEM
            fig = mne.viz.plot_bem(subject=fs_sub, 
                                subjects_dir=fs_sub_dir,
                                orientation='coronal', 
                                brain_surfaces='white',
                                show=False)
            fig.savefig(bem_figname)

            print("Reading info")
            epoched_fname = 'sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_epo.fif'
            epoched_fif = op.join(epoched_dir, epoched_fname)
                                
            info = mne.read_epochs(epoched_fif, preload=True, verbose=True).info  # one 7min50sec epochs

            print("Setting up the coregistration model")
            fiducials = "estimated"  # gets fiducials from fsaverage
            coreg = mne.coreg.Coregistration(info, 
                                            subject=fs_sub, 
                                            subjects_dir=fs_sub_dir,
                                            fiducials=fiducials)
            tr1 = coreg.trans

            fig = mne.viz.plot_alignment(info, 
                                        trans=coreg.trans, 
                                        **plot_kwargs)

            print("Initial fit with fiducials")
            """ firstly fit with 3 fiducial points. This allows to find a good
            initial solution before optimization using head shape points"""
            coreg.fit_fiducials(verbose=True)
            tr2 = coreg.trans
            #assert(np.all(tr1 == tr2))  # checked if the trans is different than before- should be and raises assertion error

            fig = mne.viz.plot_alignment(info, 
                                        trans=coreg.trans, 
                                        **plot_kwargs)

            print("Refining with ICP")
            """ secondly we refine the transformation using a few iterations of the
            Iterative Closest Point (ICP) algorithm."""
            coreg.fit_icp(n_iterations=20, 
                        nasion_weight=1., 
                        verbose=True)
            tr3 = coreg.trans
            # assert(np.all(tr2 == tr3))

            fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)

            print("Omitting bad points")
            """ we now remove the points that are not on the scalp"""
            coreg.omit_head_shape_points(distance=5/1000)  # distance is in meters- try smaller distances
            # dig_dict_after_omit_head_points = coreg._filtered_extra_points # save the list of digitalised points in coreg
            tr4 = coreg.trans
            assert(np.all(tr3 == tr4))   # checked if the trans is different than before- should not be and no error

            fig = mne.viz.plot_alignment(info, 
                                        trans=coreg.trans, 
                                        **plot_kwargs)

            print("Final coregistration fit")
            coreg.fit_icp(n_iterations=20, 
                        nasion_weight=10., 
                        verbose=True)
            # dig_info_after_final_fit_icp = coreg._info["dig"]
            # dig_dict_after_final_fit_icp = coreg._dig_dict["hsp"][coreg._extra_points_filter]  # save the list of digitalised points in coreg
            tr5 = coreg.trans
            # assert(np.all(tr4 ~= tr5))  # checked if the trans is different than before- should be and raises assertion error
            
            """this shows the coregistration process is being done correctly, the visualisation doesn't
            take into account the head shape points that are excluded and therefore show all of them. This isn't
            an issue with coregistration."""

            coreg_fig = mne.viz.plot_alignment(info, 
                                            trans=coreg.trans, 
                                            **plot_kwargs)
            mne.viz.set_3d_view(coreg_fig, **view_kwargs)

            # To save the fig above, take a screenshot of the 3D scene
            screenshot = coreg_fig.plotter.screenshot()

            # Compute distance between MRI and HSP
            dists = coreg.compute_dig_mri_distances() * 1e3  # in mm
            text = f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm \
                \n{np.min(dists):.2f} mm / \n{np.max(dists):.2f} mm"

            # The screenshot is just a NumPy array, so we can display it via imshow()
            # and then save it to a file.
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(screenshot, origin='upper')
            ax.set_axis_off()  # Disable axis labels and ticks
            fig.text(text)
            fig.tight_layout()
            fig.savefig(coreg_figname, dpi=150)

            print(f"Saving trans of subject # {i}")
            mne.write_trans(trans_fname, 
                            coreg.trans,
                            overwrite=True)
        except:
            print(f'an error occured while reading subject # {subjectID} - moving on to next subject')
            pass

elif coreg == 'manual':

    subjectID = '120309'  # FreeSurfer subject name - 120469  120462  120309
    fs_sub = f'sub-CC{subjectID}_T1w'  # name of fs folder for each subject

    # Specify specific file names
    fs_sub_dir = op.join(rds_dir, f'cc700/mri/pipeline/release004/BIDS_20190411/anat')  # FreeSurfer directory (after running recon all)
    deriv_folder = op.join(rds_dir, 'derivatives/meg/source/freesurfer', fs_sub[:-4])
    
    if not os.path.exists(deriv_folder):
        os.makedirs(deriv_folder)
    trans_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + trans_suffix + meg_extension)
    bem_fname = trans_fname.replace(trans_suffix, bem_suffix)  
    bem_figname = bem_fname.replace(meg_extension, '.png')
    coreg_figname = bem_figname.replace(bem_suffix, 'final_coreg')

    # First inspect the surface reconstruction
    Brain = mne.viz.get_brain_class()

    brain = Brain(subject=fs_sub, 
                hemi='lh', 
                surf='pial',
                subjects_dir=fs_sub_dir, 
                size=(800, 600))

    brain.add_annotation('aparc.a2009s', borders=False)
    # Get Boundary Element model (BEM) solution
    """ run this section after the watershed_bem surfaces are read in freesurfer,
    (using my_recon.sh batch script)"""

    # Creat BEM model
    conductivity = (.3,)  # for single layer
    model = mne.make_bem_model(subject=fs_sub, 
                            subjects_dir=fs_sub_dir,
                            ico=4, 
                            conductivity=conductivity)

    # BEM solution is derived from the BEM model
    bem = mne.make_bem_solution(model)
    mne.write_bem_solution(bem_fname, 
                        bem, 
                        overwrite=True, 
                        verbose=True)
    # Visualize the BEM
    fig = mne.viz.plot_bem(subject=fs_sub, 
                        subjects_dir=fs_sub_dir,
                        orientation='coronal', 
                        brain_surfaces='white',
                        show=False)
    fig.savefig(bem_figname)

    ### MANUAL COREGISTRATION ##
    """ manually pick the fiducials and coregister MEG with MRI.
    for instructions check out:https://www.youtube.com/watch?v=ALV5qqMHLlQ""" 
    mne.gui.coregistration(subject=fs_sub, subjects_dir=fs_sub_dir)

    # Use this for info path in the gui
    info_fname = '/Volumes/quinna-camcan/derivatives/meg/sensor/epoched-7min50/sub-CC{subjectID}_ses-rest_task-rest_megtransdef_epo.fif'
    trans_fname = '/Volumes/quinna-camcan/derivatives/meg/source/freesurfer/sub-CC{subjectID}/sub-CC{subjectID}_coreg-trans.fif'
    fiducials_fname = op.join(fs_sub_dir, fs_sub, 'bem', fs_sub + '-fiducials.fif')

