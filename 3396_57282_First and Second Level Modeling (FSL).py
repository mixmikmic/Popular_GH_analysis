import nipype.algorithms.modelgen as model   # model generation
from  nipype.interfaces import fsl, ants      
from nipype.interfaces.base import Bunch
import os,json,glob,sys
import numpy
import nibabel
import nilearn.plotting

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

datadir='/home/vagrant/data/ds000114_R2.0.1/'
    
results_dir = os.path.abspath("../../results")
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

from nipype.caching import Memory
mem = Memory(base_dir='.')

print('Using data from',datadir)

from bids.grabbids import BIDSLayout
layout = BIDSLayout(datadir)
layout.get(type="bold", task="fingerfootlips", session="test", extensions="nii.gz")[0].filename

import pandas as pd
events = pd.read_csv(os.path.join(datadir, "task-fingerfootlips_events.tsv"), sep="\t")
events

for trial_type in events.trial_type.unique():
    print(events[events.trial_type == trial_type])

events[events.trial_type == 'Finger'].duration

source_epi = layout.get(type="bold", task="fingerfootlips", session="test", extensions="nii.gz")[5]

confounds = pd.read_csv(os.path.join(datadir, "derivatives", "fmriprep", 
                                        "sub-%s"%source_epi.subject, "ses-%s"%source_epi.session, "func", 
                                        "sub-%s_ses-%s_task-fingerfootlips_bold_confounds.tsv"%(source_epi.subject,
                                                                                                                           source_epi.session)),
           sep="\t", na_values="n/a")

info = [Bunch(conditions=['Finger',
                          'Foot',
                          'Lips'],
              onsets=[list(events[events.trial_type == 'Finger'].onset-10),
                      list(events[events.trial_type == 'Foot'].onset-10),
                      list(events[events.trial_type == 'Lips'].onset-10)],
              durations=[list(events[events.trial_type == 'Finger'].duration),
                          list(events[events.trial_type == 'Foot'].duration),
                          list(events[events.trial_type == 'Lips'].duration)],
             regressors=[list(confounds.FramewiseDisplacement.fillna(0)[4:]),
                         list(confounds.aCompCor0[4:]),
                         list(confounds.aCompCor1[4:]),
                         list(confounds.aCompCor2[4:]),
                         list(confounds.aCompCor3[4:]),
                         list(confounds.aCompCor4[4:]),
                         list(confounds.aCompCor5[4:]),
                        ],
             regressor_names=['FramewiseDisplacement',
                              'aCompCor0',
                              'aCompCor1',
                              'aCompCor2',
                              'aCompCor3',
                              'aCompCor4',
                              'aCompCor5',])
       ]

skip = mem.cache(fsl.ExtractROI)
skip_results = skip(in_file=os.path.join(datadir, "derivatives", "fmriprep", 
                                        "sub-%s"%source_epi.subject, "ses-%s"%source_epi.session, "func", 
                                        "sub-%s_ses-%s_task-fingerfootlips_bold_space-MNI152NLin2009cAsym_preproc.nii.gz"%(source_epi.subject,
                                                                                                                           source_epi.session)),
                     t_min=4, t_size=-1)

s = model.SpecifyModel()
s.inputs.input_units = 'secs'
s.inputs.functional_runs = skip_results.outputs.roi_file
s.inputs.time_repetition = layout.get_metadata(source_epi.filename)["RepetitionTime"]
s.inputs.high_pass_filter_cutoff = 128.
s.inputs.subject_info = info
specify_model_results = s.run()
s.inputs

finger_cond = ['Finger','T', ['Finger'],[1]]
foot_cond = ['Foot','T', ['Foot'],[1]]
lips_cond = ['Lips','T', ['Lips'],[1]]
lips_vs_others = ["Lips vs. others",'T', ['Finger', 'Foot', 'Lips'],[-0.5, -0.5, 1]]
all_motor = ["All motor", 'F', [finger_cond, foot_cond, lips_cond]]
contrasts=[finger_cond, foot_cond, lips_cond, lips_vs_others, all_motor]
           
level1design = mem.cache(fsl.model.Level1Design)
level1design_results = level1design(interscan_interval = layout.get_metadata(source_epi.filename)["RepetitionTime"],
                                    bases = {'dgamma':{'derivs': True}},
                                    session_info = specify_model_results.outputs.session_info,
                                    model_serial_correlations=True,
                                    contrasts=contrasts)

level1design_results.outputs

modelgen = mem.cache(fsl.model.FEATModel)
modelgen_results = modelgen(fsf_file=level1design_results.outputs.fsf_files,
                            ev_files=level1design_results.outputs.ev_files)
modelgen_results.outputs

desmtx=numpy.loadtxt(modelgen_results.outputs.design_file,skiprows=5)
plt.imshow(desmtx,aspect='auto',interpolation='nearest',cmap='gray')

cc=numpy.corrcoef(desmtx.T)
plt.imshow(cc,aspect='auto',interpolation='nearest', cmap=plt.cm.viridis)
plt.colorbar()

mask = mem.cache(fsl.maths.ApplyMask)
mask_results = mask(in_file=skip_results.outputs.roi_file,
                    mask_file=os.path.join(datadir, "derivatives", "fmriprep", 
                                        "sub-%s"%source_epi.subject, "ses-%s"%source_epi.session, "func", 
                                        "sub-%s_ses-%s_task-fingerfootlips_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz"%(source_epi.subject,
                                                                                                                             source_epi.session)))
mask_results.outputs

filmgls= mem.cache(fsl.FILMGLS)
filmgls_results = filmgls(in_file=mask_results.outputs.out_file,
                          design_file = modelgen_results.outputs.design_file,
                          tcon_file = modelgen_results.outputs.con_file,
                          fcon_file = modelgen_results.outputs.fcon_file,
                          autocorr_noestimate = True)
filmgls_results.outputs

for t_map in filmgls_results.outputs.zstats:
    nilearn.plotting.plot_glass_brain(nilearn.image.smooth_img(t_map, 8),
                                      display_mode='lyrz', colorbar=True, plot_abs=False, threshold=2.3)

for t_map in [filmgls_results.outputs.zfstats]:
    nilearn.plotting.plot_glass_brain(nilearn.image.smooth_img(t_map, 8),
                                      display_mode='lyrz', colorbar=True, plot_abs=False, threshold=2.3)

for t_map in filmgls_results.outputs.copes:
    nilearn.plotting.plot_glass_brain(nilearn.image.smooth_img(t_map, 8),
                                      display_mode='lyrz', colorbar=True, plot_abs=False, vmax=30)

for t_map in filmgls_results.outputs.tstats:
    nilearn.plotting.plot_stat_map(nilearn.image.smooth_img(t_map, 8), colorbar=True, threshold=2.3)

copes = {}
for i in range(10):
    source_epi = layout.get(type="bold", task="fingerfootlips", session="test", extensions="nii.gz")[i]

    confounds = pd.read_csv(os.path.join(datadir, "derivatives", "fmriprep", 
                                            "sub-%s"%source_epi.subject, "ses-%s"%source_epi.session, "func", 
                                            "sub-%s_ses-%s_task-fingerfootlips_bold_confounds.tsv"%(source_epi.subject,
                                                                                                                               source_epi.session)),
               sep="\t", na_values="n/a")

    info = [Bunch(conditions=['Finger',
                              'Foot',
                              'Lips'],
                  onsets=[list(events[events.trial_type == 'Finger'].onset-10),
                          list(events[events.trial_type == 'Foot'].onset-10),
                          list(events[events.trial_type == 'Lips'].onset-10)],
                  durations=[list(events[events.trial_type == 'Finger'].duration),
                              list(events[events.trial_type == 'Foot'].duration),
                              list(events[events.trial_type == 'Lips'].duration)],
                 regressors=[list(confounds.FramewiseDisplacement.fillna(0)[4:]),
                             list(confounds.aCompCor0[4:]),
                             list(confounds.aCompCor1[4:]),
                             list(confounds.aCompCor2[4:]),
                             list(confounds.aCompCor3[4:]),
                             list(confounds.aCompCor4[4:]),
                             list(confounds.aCompCor5[4:]),
                            ],
                 regressor_names=['FramewiseDisplacement',
                                  'aCompCor0',
                                  'aCompCor1',
                                  'aCompCor2',
                                  'aCompCor3',
                                  'aCompCor4',
                                  'aCompCor5',])
           ]

    skip = mem.cache(fsl.ExtractROI)
    skip_results = skip(in_file=os.path.join(datadir, "derivatives", "fmriprep", 
                                            "sub-%s"%source_epi.subject, "ses-%s"%source_epi.session, "func", 
                                            "sub-%s_ses-%s_task-fingerfootlips_bold_space-MNI152NLin2009cAsym_preproc.nii.gz"%(source_epi.subject,
                                                                                                                               source_epi.session)),
                         t_min=4, t_size=-1)

    s = model.SpecifyModel()
    s.inputs.input_units = 'secs'
    s.inputs.functional_runs = skip_results.outputs.roi_file
    s.inputs.time_repetition = layout.get_metadata(source_epi.filename)["RepetitionTime"]
    s.inputs.high_pass_filter_cutoff = 128.
    s.inputs.subject_info = info
    specify_model_results = s.run()
    
    finger_cond = ['Finger','T', ['Finger'],[1]]
    foot_cond = ['Foot','T', ['Foot'],[1]]
    lips_cond = ['Lips','T', ['Lips'],[1]]
    lips_vs_others = ["Lips vs. others",'T', ['Finger', 'Foot', 'Lips'],[-0.5, -0.5, 1]]
    all_motor = ["All motor", 'F', [finger_cond, foot_cond, lips_cond]]
    contrasts=[finger_cond, foot_cond, lips_cond, lips_vs_others, all_motor]

    level1design = mem.cache(fsl.model.Level1Design)
    level1design_results = level1design(interscan_interval = layout.get_metadata(source_epi.filename)["RepetitionTime"],
                                        bases = {'dgamma':{'derivs': True}},
                                        session_info = specify_model_results.outputs.session_info,
                                        model_serial_correlations=True,
                                        contrasts=contrasts)
    
    modelgen = mem.cache(fsl.model.FEATModel)
    modelgen_results = modelgen(fsf_file=level1design_results.outputs.fsf_files,
                                ev_files=level1design_results.outputs.ev_files)
    
    mask = mem.cache(fsl.maths.ApplyMask)
    mask_results = mask(in_file=skip_results.outputs.roi_file,
                        mask_file=os.path.join(datadir, "derivatives", "fmriprep", 
                                        "sub-%s"%source_epi.subject, "ses-%s"%source_epi.session, "func", 
                                        "sub-%s_ses-%s_task-fingerfootlips_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz"%(source_epi.subject,
                                                                                                                             source_epi.session)))
    
    filmgls= mem.cache(fsl.FILMGLS)
    filmgls_results = filmgls(in_file=mask_results.outputs.out_file,
                              design_file = modelgen_results.outputs.design_file,
                              tcon_file = modelgen_results.outputs.con_file,
                              fcon_file = modelgen_results.outputs.fcon_file,
                              autocorr_noestimate = True)
                                                                                                                             
    copes[source_epi.subject] = list(filmgls_results.outputs.copes)                                                                                                                         


smooth_copes = []
for k,v in copes.items():
    smooth_cope = nilearn.image.smooth_img(v[3], 8)
    smooth_copes.append(smooth_cope)
    nilearn.plotting.plot_glass_brain(smooth_cope,
                                      display_mode='lyrz', 
                                      colorbar=True, 
                                      plot_abs=False, 
                                      vmax=30)

nilearn.plotting.plot_glass_brain(nilearn.image.mean_img(smooth_copes),
                                  display_mode='lyrz', 
                                  colorbar=True, 
                                  plot_abs=False)

brainmasks = glob.glob(os.path.join(datadir, "derivatives", "fmriprep", "sub-*", "ses-test", "func", "*task-fingerfootlips_*space-MNI152NLin2009cAsym*_brainmask.nii*"))

for mask in brainmasks:
    nilearn.plotting.plot_roi(mask)
    
mean_mask = nilearn.image.mean_img(brainmasks)
nilearn.plotting.plot_stat_map(mean_mask)
group_mask = nilearn.image.math_img("a>=0.95", a=mean_mask)
nilearn.plotting.plot_roi(group_mask)

get_ipython().system('mkdir -p {datadir}/derivatives/custom_modelling/')

copes_concat = nilearn.image.concat_imgs(smooth_copes, auto_resample=True)
copes_concat.to_filename(os.path.join(datadir, "derivatives", "custom_modelling", "lips_vs_others_copes.nii.gz"))

group_mask = nilearn.image.resample_to_img(group_mask, copes_concat, interpolation='nearest')
group_mask.to_filename(os.path.join(datadir, "derivatives", "custom_modelling", "group_mask.nii.gz"))

group_mask.shape

randomise = mem.cache(fsl.Randomise)
randomise_results = randomise(in_file=os.path.join(datadir, "derivatives", "custom_modelling", "lips_vs_others_copes.nii.gz"),
                              mask=os.path.join(datadir, "derivatives", "custom_modelling", "group_mask.nii.gz"),
                              one_sample_group_mean=True,
                              tfce=True,
                              vox_p_values=True,
                              num_perm=500)
randomise_results.outputs

nilearn.plotting.plot_stat_map(randomise_results.outputs.t_corrected_p_files[0], threshold=0.95)

fig = nilearn.plotting.plot_stat_map(randomise_results.outputs.tstat_files[0], alpha=0.5, cut_coords=(-21, 0, 18))
fig.add_contours(randomise_results.outputs.t_corrected_p_files[0], levels=[0.95], colors='w')

get_ipython().magic('pinfo nilearn.plotting.plot_stat_map')



