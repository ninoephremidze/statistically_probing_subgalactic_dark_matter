import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import matplotlib as mpl
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LightModel.light_model import LightModel
from multiprocessing import Pool
import dynesty
import argparse
import pickle
import time

from util_funcs import *

def main():

    ##################################################################################
    #                                 Set parameters                                 #
    ##################################################################################

    parser = argparse.ArgumentParser()

    parser.add_argument("--fit_mocks_sampling", action="store_true",
        help="Fit mocks with dynesty sampling.")

    parser.add_argument("--plot_figs", action="store_true", default=True,
        help="Enable plotting of figures (default: True).")

    parser.add_argument("--no_plot_figs", action="store_false", dest="plot_figs",
        help="Disable plotting of figures.")

    parser.add_argument("--save_figs", action="store_true",
        help="Save generated plots.")

    parser.add_argument("--dynesty_analysis", action="store_true",
        help="Run dynesty analysis on saved results instead of fitting.")

    parser.add_argument("--cab_model", type=str, default="CURVED_ARC_SIS_MST",
        help="CAB model to be fitted with sampling.")

    parser.add_argument("--exp_time", type=float, default=1200,
        help="JWST exposure time in seconds.")

    parser.add_argument("--num_pix", type=int, default=100,
        help="Side length of square mock images in pixels.")

    parser.add_argument("--base_seed", type=int, default=123,
        help="Base seed for reproducibility (controls source pos & COSMOS source).")

    parser.add_argument("--num_mocks_sets", type=int, default=100,
        help="Number of mock sets to generate (batch of 100 as default).")

    parser.add_argument("--parallel_cores", type=int, default=0,
        help="Number of cores to use for multiprocessing (default auto, 0 to disable).")

    # Cluster lens parameters
    parser.add_argument("--M200", type=float, default=1e15,
        help="Cluster halo mass M200 (Msun).")

    parser.add_argument("--c_conc", type=float, default=4.0,
        help="Cluster concentration parameter c.")

    parser.add_argument("--z_lens", type=float, default=0.4,
        help="Lens redshift.")

    parser.add_argument("--z_source", type=float, default=1.0,
        help="Source redshift.")

    parser.add_argument("--offset", type=float, default=0.3,
        help="Minimum distance of source from caustic curve (arcsec).")

    parser.add_argument("--width", type=float, default=0.6,
        help="Width of strip between caustics to place source (arcsec).")

    # Substructure parameters
    parser.add_argument("--substructure_model", type=str, default="NONE",
        help="Dark matter model for substructure (CDM, WDM, SIDM, ULDM, or NONE).")
    
    parser.add_argument("--log10_m_uldm", type=float, default=None,
        help="log base 10 of the ULDM particle mass (eV).")

    parser.add_argument("--log10_dNdA", type=float, default=0.0,
        help="Amplitude of projected SHMF (Natarajan+2017 Fig. 4).")

    parser.add_argument("--log_mhigh", type=float, default=8.0,
        help="Upper bound on subhalo mass rendered with pyHalo (log10).")

    args = parser.parse_args()
    
    fit_mocks_sampling = args.fit_mocks_sampling
    plot_figs = args.plot_figs
    save_figs = args.save_figs
    save_mocks = True
    dynesty_analysis = args.dynesty_analysis

    cab_model = args.cab_model           
                # CAB model to be fitted with sampling
    exp_time = args.exp_time                            
                # JWST exposure time in seconds
    num_pix = args.num_pix                              
                # Side length of square mock images in pixels
    base_seed = args.base_seed                            
                # Base seed for reproducibility (-> source position, COSMOS source)
    num_mocks_sets = args.num_mocks_sets
                # Number of mock sets to be generated (3 lensed mocks in each set)

    parallel_cores = args.parallel_cores if args.parallel_cores is not None else min(112, (num_pix // 10) * (num_pix // 10))
                # 0 to turn off multi-processing, positive int to specify cores

    # Cluster lens parameters

    M200, c_conc = args.M200, args.c_conc                   
                # Mass and concentration of cluster, from CDM mass-c relations 
    z_lens, z_source = args.z_lens, args.z_source                
                # Lens and source redshift
    offset, width = args.offset, args.width                  
                # Source placed between [offset, width] arcseconds from caustic curve

    # Substructure parameters

    substructure_model = args.substructure_model
                # Dark matter model used to generate substructure (CDM, WDM, SIDM, or ULDM)
                # If NONE, smooth mocks with no substructure are generates
    log10_m_uldm = args.log10_m_uldm
                # log base 10 of the ULDM particle mass
    log10_dNdA = args.log10_dNdA
                # Amplitude of projected SHMF estimated from Natarajan et al. 2017 (Figure 4)
    log_m_host = np.log10(M200)
                # Substructure generated taking cluster halo as parant halo
    log_mhigh = args.log_mhigh
                # Upper bound on the subhalo mass rendered with pyHalo

    out_dir = f'M=1e{int(np.log10(M200))}_c={int(c_conc)}_zl={z_lens}_zs={z_source}_o={offset}_w={width}_npix={num_pix}_exp={int(exp_time/60)}'
    
    if substructure_model == "ULDM":
        
        out_dir += f'_uldm_m={log10_m_uldm}'
        
    os.makedirs(out_dir, exist_ok=True)
                # Directory where mock image files will be saved

    ##################################################################################
    #                     Toy Cluster Lens Model: Elliptical NFW                     #
    ##################################################################################

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    lens_cosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=cosmo)

    Rs_angle, alpha_Rs = lens_cosmo.nfw_physical2angle(M=M200, c=c_conc)

    lens_model_list_macro = ['NFW_ELLIPSE_CSE']
    lens_model_macro = LensModel(lens_model_list_macro)
    lens_ext_macro = LensModelExtensions(lens_model_macro)

    kwargs_lens_macro = [{
        'Rs': Rs_angle,
        'alpha_Rs': alpha_Rs,
        'e1': 0.27,   # Average cluster ellipticity from Shin et al. 2018
        'e2': 0.0,
        'center_x': 0.0,
        'center_y': 0.0}]

    for mock_set_idx in range(num_mocks_sets):

        ##################################################################################
        #                                   COSMOS Source                                #
        ##################################################################################

        base_seed += 1
        random.seed(base_seed); np.random.seed(base_seed)

        # Generate a random COSMOS source, smoothing out HST noise with a Gaussian
        src_list, kwargs_source = get_source_kwargs(base_seed, z_source, smoothing_sigma=0.1)
        src_model = LightModel(light_model_list=src_list)

        # Generate a random source position between two caustic lines
        (x_src, y_src, d,
         caustic_outer, caustic_inner, bbox, 
         ra_crit, dec_crit, ra_caus, dec_caus) = generate_source_pos(lens_ext_macro, kwargs_lens_macro,
                                                                     offset, width, base_seed)

        # Ray trace source position to find locations of 3 lensed images 
        solver = LensEquationSolver(lens_model_macro)
        x_imgs, y_imgs = solver.image_position_from_source(x_src, y_src, kwargs_lens_macro,
                                                           search_window=30.)

        if plot_figs == True:
            visualize_source_selection(lens_model_macro, kwargs_lens_macro, caustic_inner, caustic_outer,
                                       ra_caus, dec_caus, ra_crit, dec_crit, bbox,
                                       offset, width, x_src, y_src, x_imgs, y_imgs, save_figs, out_dir, mock_set_idx)

        ##################################################################################
        #                             Mock image generation                              #
        ##################################################################################

        (lensed_imgs, noises, 
         mock_imgs, pixel_errors, 
         all_models_list, all_lenses_list) = generate_mock_set(cosmo, kwargs_lens_macro, kwargs_source,
                                                               Rs_angle, alpha_Rs,
                                                               x_imgs, y_imgs, z_lens, z_source,
                                                               substructure_model,
                                                               log10_dNdA, log_m_host, log_mhigh,
                                                               mock_set_idx, base_seed, exp_time, num_pix, out_dir,
                                                               parallel_cores = parallel_cores,
                                                               save_mocks = save_mocks,
                                                               plot_figs = plot_figs,
                                                               save_figs = save_figs,
                                                               log10_m_uldm = log10_m_uldm)

        ##################################################################################
        #                           Estimate CAB params from NFW                         #
        ##################################################################################

        cab_lensed_imgs = []
        all_cab_estimates = []

        for i in range(len(x_imgs)):

            cab_lensed_img, all_cab_estimates = render_cab_est_imgs(i, lens_ext_macro,
                                                                    x_imgs, y_imgs, 
                                                                    kwargs_source, kwargs_lens_macro,
                                                                    z_lens, z_source, cosmo, 
                                                                    all_cab_estimates, num_pix, exp_time)
            cab_lensed_imgs.append(cab_lensed_img)

        if plot_figs == True:

            visualize_cab_est_fit(substructure_model, mock_imgs, cab_lensed_imgs, pixel_errors,
                                  save_figs, exp_time, out_dir, mock_set_idx)

        if save_mocks:

            np.save(f"{out_dir}/subs={substructure_model}_set_#{mock_set_idx+1}_seed={base_seed}_cab_est_model.npy", cab_lensed_imgs)

    #     ##################################################################################
    #     #                         Fit mocks with dynesty sampling                        #
    #     ##################################################################################

    #     cab_bestf_imgs = []

    #     for i in range(len(x_imgs)):

    #         kwargs_estimate = lens_ext_macro.curved_arc_estimate(x_imgs[i], y_imgs[i], kwargs_lens_macro)
    #         prior_width = 1.

    #         maxparam, minparam, npar = set_up_priors(kwargs_estimate, prior_width, cab_model)

    #         if fit_mocks_sampling == True:

    #             pool = Pool(nparal)

    #             logl_args = (i, cab_model, kwargs_source, 
    #                          z_lens, z_source, cosmo, exp_time,
    #                          mock_imgs, pixel_errors, num_pix)
    #             ptform_args = (minparam, maxparam)

    #             dsampler = dynesty.NestedSampler(lnlike, ptform, npar, pool=pool, queue_size=nparal,
    #                                             logl_args=logl_args, ptform_args=ptform_args)
    #             dsampler.run_nested(dlogz = 0.001)

    #             dresults = dsampler.results

    #             save_obj(dresults, f"{out_dir}/subs={substructure_model}_{cab_model}_set#{mock_set_idx}_img#{i+1}_prior_width={prior_width}_dynesty_fit")

    #         elif dynesty_analysis == True:

    #             dresults = load_obj(f"{out_dir}/subs={substructure_model}_{cab_model}_set#{mock_set_idx}_img#{i+1}_prior_width={prior_width}_dynesty_fit")

    #             lnl, bestfs, errs = bestfit(dresults)

    #             make_dynesty_cornerplot(dresults, i, cab_model, npar, bestfs, all_cab_estimates, exp_time)

    #             cab_bestf_img = render_cab_bestf_imgs(cab_model, kwargs_source, bestfs, z_source, z_lens, cosmo, num_pix, exp_time)

    #             cab_bestf_imgs.append(cab_bestf_img)


    #     if plot_figs == True and dynesty_analysis == True:

    #         visualize_cab_dyn_fit(mock_imgs, cab_bestf_imgs, pixel_errors, save_figs, exp_time)

if __name__ == "__main__":
    
    main()