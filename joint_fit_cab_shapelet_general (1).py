import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import matplotlib
from lenstronomy.Util import util
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Plots import lens_plot, plot_util
from lenstronomy.Util import simulation_util as sim_util
from lenstronomy.Util import param_util, image_util
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
import lenstronomy.Util.param_util as param_util
from multiprocessing import Pool
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit
import lenstronomy.ImSim.de_lens as de_lens
import dynesty
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot
# Import PySwarms
import pyswarms as ps
import copy
import pickle
from astropy.cosmology import Planck15 as cosmo
import argparse
from util_funcs import *

########################################################

parser = argparse.ArgumentParser()
parser.add_argument("--cab_model", default="CURVED_ARC_SIS_MST")
parser.add_argument("--mock_set_idx", type=int, default=0)
parser.add_argument("--nmax", type=int, default=14)
parser.add_argument("--exp_time", type=int, default=1200)
parser.add_argument("--substructure_model", type=str, default="NONE")
parser.add_argument("--log10_m_uldm", type=float, default=-21.0)

args = parser.parse_args()

cab_model = args.cab_model
mock_set_idx = args.mock_set_idx
nmax = args.nmax
exp_time = args.exp_time
substructure_model=args.substructure_model
log10_m_uldm=args.log10_m_uldm

########################################################
base_seed = 124 + mock_set_idx
num_pix = 100 #200
filt = 150
cnk = 1

deltaPix = 0.031230659851709842*cnk  #  pixel size in arcsec (area per pixel = deltaPix**2)
M200, c_conc = 1e15, 4.0
z_lens, z_source = 0.4, 1.0
offset, width = 0.3, 0.6

########################################################

sigma_bkg = background_rms_per_sec(exp_time)

out_dir = f'M=1e{int(np.log10(M200))}_c={int(c_conc)}_zl={z_lens}_zs={z_source}_o={offset}_w={width}_npix={num_pix}_exp={int(exp_time/60)}'

if substructure_model == "ULDM":

    out_dir += f'_uldm_m={log10_m_uldm}'
        
file_path = f'/n/netscratch/dvorkin_lab/Lab/nephremidze/1-Statistical_substructure/0-joint_source_lens_fit/dm_substructure/{out_dir}/'

sci_stack = np.load(os.path.join(file_path + f'subs={substructure_model}_set_#{mock_set_idx+1}_seed={base_seed}_mock_imgs.npy'))
err_stack = np.load(os.path.join(file_path + f'subs={substructure_model}_set_#{mock_set_idx+1}_seed={base_seed}_mock_errs.npy'))

array200sci  = sci_stack[0]
array200err  = err_stack[0]

array200sci2 = sci_stack[1]
array200err2 = err_stack[1]

array200sci3 = sci_stack[2]
array200err3 = err_stack[2]

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 80}
matplotlib.rc('font', **font)
#rcnew = {"mathtext.fontset" : "stix"}
#plt.rcParams.update(rcnew)

#matplotlib.rcParams["axes.labelpad"] = 100.
matplotlib.rcParams['axes.linewidth'] = 2.5

#errorfixin
for i in range(len(array200err)):
    for j in range(len(array200err)):
        if array200err[i,j] == 0:
            array200err[i,j] = np.inf
        if array200err2[i,j] == 0:
            array200err2[i,j] = np.inf
        if array200err3[i,j] == 0:
            array200err3[i,j] = np.inf

def rechunk(array2d,nchunk):
    shp = np.shape(array2d)
    shpnew = [int(q/nchunk) for q in shp]
    arraynew = np.zeros(shpnew)

    for i in range(shpnew[0]):
        for j in range(shpnew[1]):
            for k in range(nchunk):
                for l in range(nchunk):
                    arraynew[i,j] += array2d[i*nchunk+k,j*nchunk+l]/(nchunk**2.) 
    return arraynew

def rechunkerr(array2d,nchunk):
    shp = np.shape(array2d)
    shpnew = [int(q/nchunk) for q in shp]
    arraynew = np.zeros(shpnew)

    for i in range(shpnew[0]):
        for j in range(shpnew[1]):
            acc = 0.
            for k in range(nchunk):
                for l in range(nchunk):
                    acc += (array2d[i*nchunk+k,j*nchunk+l]/(nchunk**2.))**2.
            arraynew[i,j] = np.sqrt(acc)
    return arraynew

arytst = rechunk(array200sci,cnk)
arytst2 = rechunk(array200sci2,cnk)
arytst3 = rechunk(array200sci3,cnk)

arytsterr = rechunkerr(array200err,cnk)
arytsterr2 = rechunkerr(array200err2,cnk)
arytsterr3 = rechunkerr(array200err3,cnk)

numPix2 = int(num_pix/cnk)

kernel_cut = np.load(f'/n/netscratch/dvorkin_lab/Lab/nephremidze/1-Statistical_substructure/sim_api_learn/jwst_psf_{filt}_sim.npy')

for i in range(91):
    for j in range(91):
        if kernel_cut[i,j] < 0:
            kernel_cut[i,j] = 0.

psf_type = 'PIXEL' #'NONE'  # 'GAUSSIAN', 'PIXEL', 

kwargs_psf = {'psf_type': psf_type, 'pixel_size': deltaPix/cnk,'kernel_point_source': kernel_cut}
#kwargs_psf = sim_util.psf_configure_simple(psf_type=psf_type, fwhm=fwhm, kernelsize=kernel_size, deltaPix=deltaPix, kernel=kernel)

psf_class = PSF(**kwargs_psf)

kwargs_numerics = {'supersampling_factor': 1}

kwargs_data2 = sim_util.data_configure_simple(numPix2, deltaPix, exp_time, sigma_bkg)
data_class2 = ImageData(**kwargs_data2)

def makemask(array,a,b,angle):
    #makes an elliptical mask of a size and angle
    shp = np.shape(array)

    likemask = np.zeros(shp,dtype=bool)
    for i in range(shp[0]):
        for j in range(shp[1]):
            xprim = np.cos(angle)*(i-shp[0]/2.) + np.sin(angle)*(j-shp[1]/2.)
            yprim = np.cos(angle)*(j-shp[1]/2.) - np.sin(angle)*(i-shp[0]/2.)
            sqrsum = (xprim/a)**2 + (yprim/b)**2. 
            if sqrsum < 1:
                likemask[i,j] = True
    return likemask

#######################################################
# small masks
likemask = np.ones((num_pix, num_pix), dtype=bool)
likemask2 = np.ones((num_pix, num_pix), dtype=bool)
likemask3 = np.ones((num_pix, num_pix), dtype=bool)



def flatten2d(arrays,likemasks):
    #flattens a set of arrays according to likemasks
    flatarr = []
    for i in range(len(likemasks)):
        for p in range(len(likemasks[i])):
            for q in range(len(likemasks[i])):
                if likemasks[i][p,q]:
                    flatarr.append(arrays[i][p,q])  
    return flatarr

def unflatten(flatimg,likemasks):
    arrays = np.zeros([len(likemasks),len(likemasks[0]),len(likemasks[0])])
    k = 0
    for i in range(len(likemasks)):
        for p in range(len(likemasks[i])):
            for q in range(len(likemasks[i])):
                if likemasks[i][p,q]:
                    arrays[i,p,q] = flatimg[k]
                    k+=1
    return arrays

# we swap places of first and second mocks, so that radial stretch of first mock = 1
imagearr = np.array([arytst,arytst2,arytst3])
noises = np.array([arytsterr,arytsterr2,arytsterr3])
likemasks = np.array([likemask,likemask2,likemask3])
likearr = likemasks

flatarr = flatten2d(imagearr,likemasks)
flaterror = flatten2d(noises,likemasks)

indices = [0,1,2]

for i in range(len(flaterror)):
    if flaterror[i] == 0:
        flaterror[i] = np.inf

unflatimg = unflatten(flatarr,likemasks)


def model_curved_fit_shapelet_sers(data,kappashear_params,source_params,likemask_list,indices):   
    A_list = []
    C_D_response_list = []
    d_list = []
    for i in indices:
    # the lens model is a supperposition of an elliptical lens model with external shear
        lens_model_list_new = ['SHIFT', cab_model]
        kwargs_lens_true_new = kappashear_params[i]

        lens_model_class = LensModel(lens_model_list=lens_model_list_new)

        source_model_list = ['SHAPELETS']
        source_model_class = LightModel(light_model_list=source_model_list)

        lensLightModel_reconstruct = LightModel([])

        data_class2.update_data(data[i])

        imageModel = ImageLinearFit(data_class=data_class2, psf_class=psf_class, kwargs_numerics=kwargs_numerics, 
                                lens_model_class=lens_model_class, source_model_class=source_model_class,
                                lens_light_model_class = lensLightModel_reconstruct,likelihood_mask=likemask_list[i])


        A = imageModel.linear_response_matrix(kwargs_lens_true_new, source_params, kwargs_lens_light=[], kwargs_ps=None)
        C_D_response, model_error = imageModel.error_response(kwargs_lens_true_new, kwargs_ps=None,kwargs_special=None)
        d = imageModel.data_response

        A_list.append(A)

        Ashp = np.shape(A)

        C_D_response_list.append(C_D_response)
        d_list.append(d)

    Ashp = np.shape(A)

    Atot = np.concatenate((A_list),axis=1)
    Ctot = np.concatenate((C_D_response_list))
    Dtot = np.concatenate((d_list))

    param, cov_param, wls_model2 = de_lens.get_param_WLS(Atot.T, 1./Ctot, Dtot, inv_bool=False)

    return wls_model2,param

ndeg = np.count_nonzero(likemasks)


###################### PRIORS #############################

all_kwargs_estimate = []

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
lens_cosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=cosmo)

Rs_angle, alpha_Rs = lens_cosmo.nfw_physical2angle(M=M200, c=c_conc)
kwargs_lens_macro = [{
    'Rs': Rs_angle,
    'alpha_Rs': alpha_Rs,
    'e1': 0.27,   # from Shin+ 2018 (MNRAS)
    'e2': 0.0,
    'center_x': 0.0,
    'center_y': 0.0}]

lens_model_macro = LensModel(['NFW_ELLIPSE_CSE'])
lens_ext         = LensModelExtensions(lens_model_macro)

random.seed(base_seed); np.random.seed(base_seed)

x_src, y_src, d, caustic_outer, caustic_inner, bbox, ra_crit, dec_crit, ra_caus, dec_caus = generate_source_pos(lens_ext, kwargs_lens_macro, offset, width, base_seed)

solver = LensEquationSolver(lens_model_macro)
x_imgs, y_imgs = solver.image_position_from_source(
    x_src, y_src, kwargs_lens_macro,
    search_window=30.)

for i in range(3):
    
    x_img, y_img = x_imgs[i], y_imgs[i]

    kwargs_estimate = lens_ext.curved_arc_estimate(x_img, y_img, kwargs_lens_macro)
    
    if i == 0:
        rad_stretch1 = kwargs_estimate['radial_stretch']
        
    # normalize the stretches by first radial stretch
    kwargs_estimate['radial_stretch'] /= rad_stretch1
    kwargs_estimate['tangential_stretch'] /= rad_stretch1

    all_kwargs_estimate.append(kwargs_estimate)

prior_width = 1.5

maxlens1, minlens1, _ = set_up_priors(all_kwargs_estimate[0], prior_width, cab_model)
maxlens1, minlens1 = maxlens1[1:], minlens1[1:]

maxlens2, minlens2, _ = set_up_priors(all_kwargs_estimate[1], prior_width, cab_model)
maxlens3, minlens3, _ = set_up_priors(all_kwargs_estimate[2], prior_width, cab_model)

maxshift = [0.1,  0.1]
minshift = [-0.1, -0.1]

maxshape = [2.5, 0.2, 0.2]
minshape = [0.0, -0.2, -0.2]

maxparam = maxlens1 + maxlens2 + maxshift + maxlens3 + maxshift + maxshape 
minparam = minlens1 + minlens2 + minshift + minlens3 + minshift + minshape

npar = len(minparam)

def lnlike(params):      
    
    if cab_model == 'CURVED_ARC_SIS_MST':

        mur1, mut1 = 1.,params[0]
        mur2, mut2 = params[3],params[4]
        mur3, mut3 = params[9],params[10]

        cv1 = params[1] #np.abs(params[1])
        cv2 = params[5] #np.abs(params[5])
        cv3 = params[11] #np.abs(params[11])

        psi_ext1 = params[2] #params[2]*np.pi - (np.sign(params[1])+1.)*np.pi/2.
        psi_ext2 = params[6] #params[6]*np.pi - (np.sign(params[5])+1.)*np.pi/2.
        psi_ext3 = params[12] #params[12]*np.pi - (np.sign(params[11])+1.)*np.pi/2.

        kwargs_kapshe = [[{'alpha_x':0.,'alpha_y':0.},
                          {'tangential_stretch': mut1, 'radial_stretch': mur1, 'curvature': cv1, 'direction': psi_ext1, 
                           'center_x': 0., 'center_y': 0.}],
                         [{'alpha_x':params[7],'alpha_y':params[8]},
                          {'tangential_stretch': mut2, 'radial_stretch': mur2, 'curvature': cv2, 'direction': psi_ext2, 
                           'center_x': 0., 'center_y': 0.}],
                         [{'alpha_x':params[13],'alpha_y':params[14]},
                          {'tangential_stretch': mut3, 'radial_stretch': mur3, 'curvature': cv3, 'direction': psi_ext3, 
                           'center_x': 0., 'center_y': 0.}]]

        source_shape = [{'n_max': nmax, 'beta': params[15]/np.power(nmax+1,0.5), 'center_x': params[16], 'center_y': params[17]}]
        
    elif cab_model == 'CURVED_ARC_SPT':

        mur1, mut1 = 1.,params[0]
        mur2, mut2 = params[5],params[6]
        mur3, mut3 = params[13],params[14]

        cv1 = params[1] #np.abs(params[1])
        cv2 = params[7] #np.abs(params[7])
        cv3 = params[15] #np.abs(params[15])

        psi_ext1 = params[2] #params[2]*np.pi - (np.sign(params[1])+1.)*np.pi/2.
        psi_ext2 = params[8] #params[8]*np.pi - (np.sign(params[7])+1.)*np.pi/2.
        psi_ext3 = params[16] #params[16]*np.pi - (np.sign(params[15])+1.)*np.pi/2.

        gamma1_mock1, gamma2_mock1 = params[3], params[4]
        gamma1_mock2, gamma2_mock2 = params[9], params[10]
        gamma1_mock3, gamma2_mock3 = params[17], params[18]

        kwargs_kapshe = [[{'alpha_x':0.,'alpha_y':0.},
                          {'tangential_stretch': mut1, 'radial_stretch': mur1, 'curvature': cv1, 'direction': psi_ext1, 
                           'gamma1': gamma1_mock1, 'gamma2': gamma2_mock1,
                           'center_x': 0., 'center_y': 0.}],
                         [{'alpha_x':params[11],'alpha_y':params[12]},
                          {'tangential_stretch': mut2, 'radial_stretch': mur2, 'curvature': cv2, 'direction': psi_ext2, 
                           'gamma1': gamma1_mock2, 'gamma2': gamma2_mock2,
                           'center_x': 0., 'center_y': 0.}],
                         [{'alpha_x':params[19],'alpha_y':params[20]},
                          {'tangential_stretch': mut3, 'radial_stretch': mur3, 'curvature': cv3, 'direction': psi_ext3, 
                           'gamma1': gamma1_mock3, 'gamma2': gamma2_mock3,
                           'center_x': 0., 'center_y': 0.}]]

        source_shape = [{'n_max': nmax, 'beta': params[21]/np.power(nmax+1,0.5), 'center_x': params[22], 'center_y': params[23]}]
        
    elif cab_model == 'CURVED_ARC_TAN_DIFF':
        
        mur1, mut1 = 1.,params[0]
        mur2, mut2 = params[4],params[5]
        mur3, mut3 = params[11],params[12]

        cv1 = params[1] #np.abs(params[1])
        cv2 = params[6] #np.abs(params[6])
        cv3 = params[13] #np.abs(params[13])

        psi_ext1 = params[2] #params[2]*np.pi - (np.sign(params[1])+1.)*np.pi/2.
        psi_ext2 = params[7] #params[7]*np.pi - (np.sign(params[6])+1.)*np.pi/2.
        psi_ext3 = params[14] #params[14]*np.pi - (np.sign(params[13])+1.)*np.pi/2.

        dtan1 = params[3]
        dtan2 = params[8]
        dtan3 = params[15]

        kwargs_kapshe = [[{'alpha_x':0.,'alpha_y':0.},
                          {'tangential_stretch': mut1, 'radial_stretch': mur1, 'curvature': cv1, 'direction': psi_ext1,
                           'dtan_dtan': dtan1,
                           'center_x': 0., 'center_y': 0.}],
                         [{'alpha_x':params[9],'alpha_y':params[10]},
                          {'tangential_stretch': mut2, 'radial_stretch': mur2, 'curvature': cv2, 'direction': psi_ext2, 
                           'dtan_dtan': dtan2,
                           'center_x': 0., 'center_y': 0.}],
                         [{'alpha_x':params[16],'alpha_y':params[17]},
                          {'tangential_stretch': mut3, 'radial_stretch': mur3, 'curvature': cv3, 'direction': psi_ext3, 
                           'dtan_dtan': dtan3,
                           'center_x': 0., 'center_y': 0.}]]

        source_shape = [{'n_max': nmax, 'beta': params[18]/np.power(nmax+1,0.5), 'center_x': params[19], 'center_y': params[20]}]

    fit,paramq = model_curved_fit_shapelet_sers(imagearr,kwargs_kapshe,source_shape,likearr,indices)

    return -0.5*np.sum((((fit-flatarr)/flaterror)**2.))


# Define our uniform prior.
def ptform(u):
    """Transforms samples `u` drawn from the unit cube to samples to those
    from our uniform prior within [-10., 10.) for each variable."""
    return minparam*(1.-u) + maxparam*u

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

nparal = 112
pool = Pool(nparal)

#"Dynamic" nested sampling.
dsampler = dynesty.NestedSampler(lnlike, ptform, npar, pool=pool, queue_size=nparal)
dsampler.run_nested(dlogz=0.001, checkpoint_file=f'{out_dir}/subs={substructure_model}_{cab_model}_shapelet_fit_set=#{mock_set_idx+1}_nmax={nmax}_prior_width={prior_width}.save')

dresults = dsampler.results

def save_obj(obj, name ):
   with open(name + '.pkl', 'wb') as f:
       pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

save_obj(dresults, f'{out_dir}/subs={substructure_model}_{cab_model}_shapelet_fit_set=#{mock_set_idx+1}_nmax={nmax}_prior_width={prior_width}')