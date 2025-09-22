import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib as mpl

import scipy.signal as signal
from scipy.spatial import cKDTree
from astropy.cosmology import FlatLambdaCDM

from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.ImSim.de_lens as de_lens
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.Util import simulation_util as sim_util
from lenstronomy.Data.psf import PSF
from lenstronomy.Data.imaging_data import ImageData
import lenstronomy.Util.util as util
from cluster_local_deflection import ClusterLocalDeflection

import copy
import pandas as pd
from paltas.Sources.cosmos import COSMOSExcludeCatalog

from JWST import *

from pyHalo.PresetModels.cdm import CDM
from pyHalo.PresetModels.uldm import ULDM
from pyHalo.Halos.tidal_truncation import TruncationBoundMassPDF

import dynesty
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot
import pickle
from multiprocessing import Pool

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.lines as lines

psf_file = '/n/netscratch/dvorkin_lab/Lab/nephremidze/1-Statistical_substructure/sim_api_learn/jwst_psf_150_sim.npy'
cosmos_folder = '/n/netscratch/dvorkin_lab/Lab/nephremidze/thesis/ML/cab/paltas/COSMOS_data/COSMOS_23.5_training_sample/'

# settings for caustic computations (for source placement)

compute_window = 30.0   # arcsec half-width (used e.g. for curves and solver search window)
grid_scale     = 0.2    # arcsec / pixel
cx, cy         = 0.0, 0.0
    
################################################################################
#                       JWST observation settings (NIRCam F150W)               #
################################################################################
 
jwst_pix = 0.031230659851709842

zp = 28.00         # zp_AB column from https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-absolute-flux-calibration-and-zeropoints#gsc.tab=0
                   # https://jwst-docs.stsci.edu/files/154689209/154689212/1/1716319320811/NRC_ZPs_1126pmap.txt

m_sky = zp - np.log10(10.26 / ((jwst_pix)**2 * 362.49))     
                    # m_sky = zp_AB - 2.5 log_10(sky counts / (1 e- / s / arcsec ^2))
                    # Total sky background = 10.26 e-/s; Area = 362.49 pixels
                    # from JWST ETC: https://jwst.etc.stsci.edu/workbook.html?wb_id=266151#

FWHM_150 = 0.05     # read from graph at https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-point-spread-functions#gsc.tab=0
                    # undrizzled
    
read_noise = 0.94   # source: https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-detector-overview/nircam-detector-performance#gsc.tab=0

ccd_gain = 2.05     # source: https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-detector-overview/nircam-detector-performance#gsc.tab=0

################################################################################
#                            COSMOS source selection                           #
################################################################################

def get_source_kwargs(base_seed, z_source, smoothing_sigma=0.):
    
    random.seed(base_seed); np.random.seed(base_seed)
    
    cosmos_folder = '/n/netscratch/dvorkin_lab/Lab/nephremidze/thesis/ML/cab/paltas/COSMOS_data/COSMOS_23.5_training_sample/'
    
    bad_list = pd.read_csv('/n/netscratch/dvorkin_lab/Lab/nephremidze/1-Statistical_substructure/bad_galaxies.csv', names=['catalog_i'])['catalog_i'].to_numpy()
    val_list = pd.read_csv('/n/netscratch/dvorkin_lab/Lab/nephremidze/1-Statistical_substructure/val_galaxies.csv', names=['catalog_i'])['catalog_i'].to_numpy()
    
    source_params = {
        'z_source': z_source, # maximum source redshift
        'cosmos_folder': cosmos_folder,
        'max_z': 1.0,
        'minimum_size_in_pixels': 120,
        'faintest_apparent_mag': 20,
        'smoothing_sigma': smoothing_sigma, #0.00,
        'random_rotation': True,
        'output_ab_zeropoint': zp,
        'min_flux_radius': 40,
        'center_x': 0.0,
        'center_y': 0.0,
        'source_exclusion_list': np.append(bad_list, val_list)}

    catalog = COSMOSExcludeCatalog('planck18', source_params)

    src_list, kwargs_source, _ = catalog.draw_source()
    
    return src_list, kwargs_source

################################################################################
#                              source position selection
################################################################################

def polygon_area(x, y):
    if (x[0] != x[-1]) or (y[0] != y[-1]):
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]
    return 0.5 * np.sum(x[:-1]*y[1:] - x[1:]*y[:-1])

def closed_path(xs, ys):
    v = np.column_stack([xs, ys])
    if not np.allclose(v[0], v[-1]):
        v = np.vstack([v, v[0]])
    codes = np.full(len(v), Path.LINETO)
    codes[0] = Path.MOVETO
    return Path(v, codes)

def identify_outer_inner_caustics(ra_caus_list, dec_caus_list):
    # loop with larger polygon areaa is outer loop 
    loops = []
    for xs, ys in zip(ra_caus_list, dec_caus_list):
        xs = np.asarray(xs); ys = np.asarray(ys)
        A = abs(polygon_area(xs, ys))
        loops.append((A, xs, ys))
    loops.sort(key=lambda t: t[0], reverse=True)
    outer = (loops[0][1], loops[0][2])
    inner = (loops[1][1], loops[1][2])
    return outer, inner

def make_caustic_kdtree(ra_caus_list, dec_caus_list):

    # P is (N, 2) array listing coordinates of each point on caustic line

    pts = []
    for xs, ys in zip(ra_caus_list, dec_caus_list):
        pts.append(np.column_stack([np.asarray(xs).ravel(), np.asarray(ys).ravel()]))
    P = np.vstack(pts)

    return cKDTree(P), P

def generate_source_pos(lens_ext, kwargs_lens, offset, width, base_seed, max_tries=50000):

    ra_crit, dec_crit, ra_caus, dec_caus = lens_ext.critical_curve_caustics(
        kwargs_lens=kwargs_lens,
        compute_window=compute_window,
        grid_scale=grid_scale,
        center_x=cx, center_y=cy)

    rng = np.random.default_rng(seed=base_seed)

    tree, _ = make_caustic_kdtree(ra_caus, dec_caus)
    (x_out, y_out), (x_in, y_in) = identify_outer_inner_caustics(ra_caus, dec_caus)
    caustic_outer = closed_path(x_out, y_out)
    caustic_inner = closed_path(x_in, y_in)

    pad = offset + width + 0.5
    x_min, x_max = x_out.min() - pad, x_out.max() + pad
    y_min, y_max = y_out.min() - pad, y_out.max() + pad

    lo, hi = offset, offset + width
    for _ in range(max_tries):
        xs = rng.uniform(x_min, x_max)
        ys = rng.uniform(y_min, y_max)
        # ensure we are inside the outer caustic but outside the inner caustic to make 3 images
        if not caustic_outer.contains_point((xs, ys)):
            continue
        if caustic_inner.contains_point((xs, ys)):
            continue
        d, _ = tree.query([xs, ys], k=1) # nearest neighbor between random point and caustic lines
        if lo <= d <= hi:
            return xs, ys, d, caustic_outer, caustic_inner, (x_min, x_max, y_min, y_max), ra_crit, dec_crit, ra_caus, dec_caus

        
##################################################################################
#                           Visualize random source pos                          #
##################################################################################

def visualize_source_selection(lens_model, kwargs_lens,
                                   caustic_inner, caustic_outer,
                                   ra_caus, dec_caus, 
                                   ra_crit, dec_crit, bbox,
                                   offset, width,
                                   x_src, y_src,
                                   x_imgs, y_imgs,
                                   save_figs,
                                   out_dir, mock_set_idx):

    nx = int(2 * compute_window / grid_scale) + 1
    x  = np.linspace(cx - compute_window/2, cx + compute_window/2, nx)
    y  = np.linspace(cy - compute_window/2, cy + compute_window/2, nx)
    xx, yy = np.meshgrid(x, y)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    x_edges = np.linspace(x[0] - dx/2, x[-1] + dx/2, nx + 1)
    y_edges = np.linspace(y[0] - dy/2, y[-1] + dy/2, nx + 1)

    mu = lens_model.magnification(xx, yy, kwargs=kwargs_lens)
    mu_abs = np.abs(mu)
    mu_log = np.full_like(mu_abs, np.nan, dtype=float)
    finite = np.isfinite(mu_abs) & (mu_abs > 0)
    mu_log[finite] = np.log10(mu_abs[finite])
    vmin = np.nanpercentile(mu_log, 5)
    vmax = np.nanpercentile(mu_log, 99)

    nx_vis, ny_vis = 300, 300
    x_min, x_max, y_min, y_max = bbox
    sx = np.linspace(x_min, x_max, nx_vis)
    sy = np.linspace(y_min, y_max, ny_vis)
    SX, SY = np.meshgrid(sx, sy)
    tree_band, _ = make_caustic_kdtree(ra_caus, dec_caus)
    dists = tree_band.query(np.column_stack([SX.ravel(), SY.ravel()]), k=1)[0].reshape(SX.shape)

    pts = np.column_stack([SX.ravel(), SY.ravel()])
    inside_outer = caustic_outer.contains_points(pts)

    if caustic_inner is not None:
        outside_inner = ~caustic_inner.contains_points(pts)
        between_mask = (inside_outer & outside_inner).reshape(SX.shape)
    else:
        between_mask = inside_outer.reshape(SX.shape)  # only one loop: “between” = inside outer

    distance_mask = (dists >= offset) & (dists <= offset + width)
    band_mask = between_mask & distance_mask

    dxs = sx[1] - sx[0]; dys = sy[1] - sy[0]
    sx_edges = np.linspace(sx[0] - dxs/2, sx[-1] + dxs/2, nx_vis + 1)
    sy_edges = np.linspace(sy[0] - dys/2, sy[-1] + dys/2, ny_vis + 1)
    band_img = np.zeros_like(dists, dtype=float); band_img[band_mask] = 1.0

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))

    pcm = axL.pcolormesh(x_edges, y_edges, mu_log, shading='auto', vmin=vmin, vmax=vmax)
    fig.colorbar(pcm, ax=axL, label=r'$\log_{10}|\mu|$')
    for xc, yc in zip(ra_crit, dec_crit):
        axL.plot(xc, yc, lw=1.6, label='Critical curve', zorder=3)
    axL.plot(np.atleast_1d(x_imgs), np.atleast_1d(y_imgs), 'o', ms=7, mec='k', mfc='cyan',
             label='Lensed images', zorder=4)
    axL.set_aspect('equal', adjustable='box')
    axL.set_xlabel('RA (")'); axL.set_ylabel('DEC (")')
    axL.set_title('Image plane')
    # axL.invert_xaxis()
    handlesL, labelsL = axL.get_legend_handles_labels()
    axL.legend(dict(zip(labelsL, handlesL)).values(), dict(zip(labelsL, handlesL)).keys(), loc='upper right')
    axL.grid(alpha=0.25)

    axR.pcolormesh(sx_edges, sy_edges, band_img, shading='auto', alpha=0.25, label='0.2"–0.4" band')
    for xs, ys in zip(ra_caus, dec_caus):
        axR.plot(xs, ys, color='k', lw=1.6, label='Caustic')
    axR.plot(caustic_outer.vertices[:,0], caustic_outer.vertices[:,1], color='k', lw=2.2)
    if caustic_inner is not None:
        axR.plot(caustic_inner.vertices[:,0], caustic_inner.vertices[:,1], color='k', lw=2.2)
    axR.plot(x_src, y_src, marker='*', ms=13, mec='k', mfc='gold',
             label=f'Selected source', zorder=5)
    axR.set_aspect('equal', adjustable='box')
    axR.set_xlabel('RA (")'); axR.set_ylabel('DEC (")')
    axR.set_title('Source plane: Caustic curves')
    handlesR, labelsR = axR.get_legend_handles_labels()
    axR.legend(dict(zip(labelsR, handlesR)).values(), dict(zip(labelsR, handlesR)).keys(), loc='upper right')
    axR.grid(alpha=0.25)

    plt.tight_layout()
    if save_figs == True:
        plt.savefig(f"{out_dir}/source_pos_#{mock_set_idx}.png")
    plt.show()
    plt.close()

##################################################################################
#                                  Plotting codes                                #
##################################################################################

def visualize_mock_set(substructure_model, lensed_imgs, noises, mock_imgs, save_figs, exp_time, out_dir, mock_set_idx):

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    im0 = axes[0][0].imshow(np.log10(lensed_imgs[0]+0.1), cmap='jet')
    axes[0][1].imshow(np.log10(lensed_imgs[1]+0.1), cmap='jet')
    axes[0][2].imshow(np.log10(lensed_imgs[2]+0.1), cmap='jet')

    im1 = axes[1][0].imshow(noises[0]+0.1, cmap='bwr')
    axes[1][1].imshow(noises[1]+0.1, cmap='bwr')
    axes[1][2].imshow(noises[2]+0.1, cmap='bwr')

    im2 = axes[2][0].imshow(np.log10(mock_imgs[0]+0.1), cmap='jet')
    axes[2][1].imshow(np.log10(mock_imgs[1]+0.1), cmap='jet')
    axes[2][2].imshow(np.log10(mock_imgs[2]+0.1), cmap='jet')

    for axes_set in axes:
        for ax in axes_set:
            ax.set_xticks([])
            ax.set_yticks([])

    axes[0][0].set_ylabel("Lensed Image", rotation=90)
    axes[1][0].set_ylabel("JWST Noise", rotation=90)
    axes[2][0].set_ylabel("Noisy Mock", rotation=90)

    plt.tight_layout()
    plt.suptitle(f"NIRCam F150W Mock: {exp_time} sec exposure", y = 1.02)
    if save_figs == True:
        plt.savefig(f"{out_dir}/jwst_mocks_subs={substructure_model}_{exp_time}s_#{mock_set_idx}.png")
    plt.show()
    plt.close()
    

def visualize_cab_est_fit(substructure_model, mock_imgs, cab_lensed_imgs, pixel_errors, save_figs, exp_time, out_dir, mock_set_idx):

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    im0 = axes[0][0].imshow(np.log10(mock_imgs[0]+0.1), cmap='jet')
    axes[0][1].imshow(np.log10(mock_imgs[1]+0.1), cmap='jet')
    axes[0][2].imshow(np.log10(mock_imgs[2]+0.1), cmap='jet')

    im1 = axes[1][0].imshow(np.log10(cab_lensed_imgs[0]+0.1), cmap='jet')
    axes[1][1].imshow(np.log10(cab_lensed_imgs[1]+0.1), cmap='jet')
    axes[1][2].imshow(np.log10(cab_lensed_imgs[2]+0.1), cmap='jet')

    res1 = (mock_imgs[0] - cab_lensed_imgs[0])/pixel_errors[0]
    res2 = (mock_imgs[1] - cab_lensed_imgs[1])/pixel_errors[1]
    res3 = (mock_imgs[2] - cab_lensed_imgs[2])/pixel_errors[2]

    im1 = axes[2][0].imshow(res1, cmap='bwr', vmin = -5., vmax = 5.)
    axes[2][1].imshow(res2, cmap='bwr', vmin = -5., vmax = 5.)
    axes[2][2].imshow(res3, cmap='bwr', vmin = -5., vmax = 5.)

    for axes_set in axes:
        for ax in axes_set:
            ax.set_xticks([])
            ax.set_yticks([])

    axes[0][0].set_ylabel("Mock Image", rotation=90)
    axes[1][0].set_ylabel("CAB Model", rotation=90)
    axes[2][0].set_ylabel("Residuals", rotation=90)

    plt.tight_layout()
    plt.suptitle(f"CAB estimate fit ({exp_time} sec exposure)", y = 1.02)
    if save_figs == True:
        plt.savefig(f"{out_dir}/cab_est_fits_subs={substructure_model}_{exp_time}s_#{mock_set_idx}.png")
    plt.show()
    plt.close()

def visualize_cab_dyn_fit(mock_imgs, cab_bestf_imgs, pixel_errors, save_figs, exp_time):

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    im0 = axes[0][0].imshow(np.log10(mock_imgs[0]+0.1), cmap='jet')
    axes[0][1].imshow(np.log10(mock_imgs[1]+0.1), cmap='jet')
    axes[0][2].imshow(np.log10(mock_imgs[2]+0.1), cmap='jet')

    im1 = axes[1][0].imshow(np.log10(cab_bestf_imgs[0]+0.1), cmap='jet')
    axes[1][1].imshow(np.log10(cab_bestf_imgs[1]+0.1), cmap='jet')
    axes[1][2].imshow(np.log10(cab_bestf_imgs[2]+0.1), cmap='jet')

    res1 = (mock_imgs[0] - cab_bestf_imgs[0])/pixel_errors[0]
    res2 = (mock_imgs[1] - cab_bestf_imgs[1])/pixel_errors[1]
    res3 = (mock_imgs[2] - cab_bestf_imgs[2])/pixel_errors[2]

    im1 = axes[2][0].imshow(res1, cmap='bwr', vmin = -5, vmax = 5)
    axes[2][1].imshow(res2, cmap='bwr', vmin = -5, vmax = 5)
    axes[2][2].imshow(res3, cmap='bwr', vmin = -5, vmax = 5)

    for axes_set in axes:
        for ax in axes_set:
            ax.set_xticks([])
            ax.set_yticks([])

    axes[0][0].set_ylabel("Mock Image", rotation=90)
    axes[1][0].set_ylabel("CAB Dynesty Best-Fit", rotation=90)
    axes[2][0].set_ylabel("Residuals", rotation=90)

    plt.tight_layout()
    plt.suptitle(f"Dynesty Fit: ({exp_time} sec exposure)", y = 1.02)
    if save_figs == True:
        plt.savefig(f"{out_dir}/cab_est_fits_{exp_time}s_#{mock_set_idx}.png")
    plt.show()
    plt.close()
    

def make_magnifications_plot(mock_set_idx, base_seed, num_pix, x_imgs, out_dir,
                             all_models_list, all_lenses_list, substructure_model,
                             parallel_cores = 0,
                             save_figs = True):

    kwargs_data_mu = make_kwargs_data_for_mu(num_pix=num_pix, jwst_pix=jwst_pix)

    mus = []

    for i in range(len(x_imgs)):
        mu_i, _, _ = parallel_magnification_total(
            num_pix=num_pix,
            kwargs_data_full=kwargs_data_mu,
            kwargs_model=all_models_list[i],
            kwargs_lens=all_lenses_list[i],
            tile_size=10,
            processes=parallel_cores)
        mus.append(mu_i)

    fig, axes = plot_mock_set_mags(mus, titles=[f"Image #{i+1}" for i in range(len(mus))], cmap="viridis")

    if save_figs:
        fig.savefig(f"{out_dir}/subs={substructure_model}_set_#{mock_set_idx+1}_seed={base_seed}_magnifications.png",
                    dpi=300, bbox_inches="tight")

##################################################################################
#                                lensing utilities                               #
##################################################################################

def generate_ith_mocks(num_pix,
                       kwargs_source, kwargs_lens, kwargs_model,
                       z_source, cosmo, exp_time,
                       *,
                       parallel_cores: int = 0):
    
    jwst_ml = JWST(jwst_pix, exp_time, zp, m_sky, FWHM_150, psf_file, read_noise, ccd_gain,
                   psf_type="PIXEL", band="NIRCam_F150W", coadd_years=None)

    kwargs_cam_obs = jwst_ml.kwargs_single_band()

    xc = yc = (num_pix - 1) / 2.0

    (_, _, ra0_def, dec0_def, _, _, M, _) = util.make_grid_with_coordtransform(
    numPix=num_pix, deltapix=jwst_pix, subgrid_res=1, left_lower=False, inverse=False)

    ra_at_xy_0  = - (M[0,0]*xc + M[0,1]*yc)
    dec_at_xy_0 = - (M[1,0]*xc + M[1,1]*yc)

    kwargs_pixel_grid = dict(
        ra_at_xy_0=ra_at_xy_0,
        dec_at_xy_0=dec_at_xy_0,
        transform_pix2angle=M,)

    kwargs_data = dict(kwargs_cam_obs)  # start from your camera/observation settings
    kwargs_data["kwargs_pixel_grid"] = kwargs_pixel_grid

    sim = SimAPI(numpix = num_pix,
                 kwargs_single_band = kwargs_data,
                 kwargs_model = kwargs_model)

    jwst_mock_class = sim.image_model_class(kwargs_numerics = {'point_source_supersampling_factor': 1})
    
    if parallel_cores > 0:
    
        psf_kernel = np.load(psf_file)

        # Ray-trace by 10x10 tiles with SimAPI, then single PSF convolution
        lensed_img, _ = image_parallel_tiles_then_psf(
            num_pix=num_pix, jwst_pix=jwst_pix,
            kwargs_data_full=kwargs_data,
            kwargs_model=kwargs_model,
            kwargs_lens=kwargs_lens,
            kwargs_source=kwargs_source,
            psf_kernel=psf_kernel,
            tile_size=10,
            processes=parallel_cores)
    else:

        lensed_img = jwst_mock_class.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source)
        
    noise = sim.noise_for_model(lensed_img)

    jwst_mock = lensed_img + noise

    sigma_fixed = sigma_map_from_data(lensed_img, kwargs_data)
    # lensed image because we only want source counts for source Poisson noise

    return lensed_img, noise, jwst_mock, sigma_fixed

def sky_rate_arcsec2_per_s(zp, m_sky):
    # e-/s/arcsec^2 for magnitude m at zeropoint zp
    return 10.0 ** (-(m_sky - zp) / 2.5)

def sigma_map_from_data(data_e_per_s, cam_kwargs):
    """
    data_e_per_s: observed image in e-/s/pixel
    Returns sigma in e-/s/pixel, same shape.
    """
    t   = float(cam_kwargs["exposure_time"])
    N   = int(cam_kwargs.get("num_exposures", 1))
    ps  = jwst_pix
    RN  = float(cam_kwargs["read_noise"])       # e- per exposure
    zp  = float(cam_kwargs["magnitude_zero_point"])
    msky= float(cam_kwargs["sky_brightness"])

    # sky: e-/s/arcsec^2 -> e-/s/pixel
    mu_sky_pix = sky_rate_arcsec2_per_s(zp, msky) * (ps**2)

    # Expected electrons from (source ≈ data) and sky, per pixel across the stack
    D_src = np.maximum(data_e_per_s, 0.0) * t * N    # e-
    Sky   = mu_sky_pix * t * N                       # e-
    # for Poisson distributions (source and sky), variance = expectation value
    Var_e = D_src + Sky + N * (RN**2)                # e-^2

    sigma = np.sqrt(np.maximum(Var_e, 0.0)) / (t * N)   # back to e-/s/pixel
    return np.maximum(sigma, 1e-6)  # small floor

def background_rms_per_sec(exp_time):
    """
    Returns background RMS (e-/s/pix) from sky + read noise only.
    """
    jwst_ml = JWST(jwst_pix, exp_time, zp, m_sky, FWHM_150, psf_file, read_noise, ccd_gain,
                   psf_type="PIXEL", band="NIRCam_F150W", coadd_years=None)

    kwargs_cam_obs = jwst_ml.kwargs_single_band()
    
    t   = exp_time
    N   = int(kwargs_cam_obs.get("num_exposures", 1))
    ps  = jwst_pix
    RN  = read_noise

    # sky rate per arcsec^2, then per pixel
    mu_sky_pix = sky_rate_arcsec2_per_s(zp, m_sky) * (ps**2)   # e-/s/pix
    Sky = mu_sky_pix * t * N                                  # e- total from sky

    Var_e = Sky + N * (RN**2)  # variance from sky + read noise
    sigma_back = np.sqrt(Var_e) / (t * N)
    return sigma_back

def get_kwargs_cab(cab_model, params):

    kwargs_cab = {'radial_stretch': params[0],
                  'tangential_stretch': params[1],
                  'curvature': params[2],
                  'direction': params[3],
                  'center_x': 0.,
                  'center_y': 0.}

    if cab_model == 'CURVED_ARC_SPT':

        kwargs_cab['gamma1'] = params[4]
        kwargs_cab['gamma2'] = params[5]

    elif cab_model == 'CURVED_ARC_TAN_DIFF':

        kwargs_cab['dtan_dtan'] = params[4]

    return kwargs_cab

##################################################################################
#                               sampling utilities                               #
##################################################################################

def set_up_priors(kwargs_estimate, prior_width, cab_model):

    prior_scales = {'radial_stretch': 1.,
                    'tangential_stretch': 1.5,
                    'curvature': 0.5,
                    'direction': np.pi/8}

    maxparam = [kwargs_estimate['radial_stretch'] + prior_width * prior_scales['radial_stretch'],
                kwargs_estimate['tangential_stretch'] + prior_width * prior_scales['tangential_stretch'],
                kwargs_estimate['curvature'] + prior_width * prior_scales['curvature'],
                kwargs_estimate['direction'] + prior_width * prior_scales['direction']]

    minparam = [kwargs_estimate['radial_stretch'] - prior_width * prior_scales['radial_stretch'],
                kwargs_estimate['tangential_stretch'] - prior_width * prior_scales['tangential_stretch'],
                kwargs_estimate['curvature'] - prior_width * prior_scales['curvature'],
                kwargs_estimate['direction'] - prior_width * prior_scales['direction']]

    if cab_model == 'CURVED_ARC_SPT':

        maxparam.extend([0.5] * 2)
        minparam.extend([-0.5] * 2)

    if cab_model == 'CURVED_ARC_TAN_DIFF':

        maxparam.extend([10.])
        minparam.extend([-10.])

    npar = len(maxparam)

    return maxparam, minparam, npar


def save_obj(obj, name ):
   with open(name + '.pkl', 'wb') as f:
       pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


        
##################################################################################
#                              dynesty fitting codes                             #
##################################################################################
        
def lnlike(params, i, cab_model, kwargs_source, z_lens, z_source, cosmo, exp_time, mock_imgs, pixel_errors, num_pix):

    kwargs_cab = get_kwargs_cab(cab_model, params)

    kwargs_model_cab = {'lens_model_list': [cab_model], 
                        'z_lens': z_lens,
                        'lens_light_model_list': [],  
                        'source_light_model_list': ['INTERPOL'],  
                        'z_source': z_source,
                        'point_source_model_list': [],
                        'cosmo': cosmo}

    kwargs_source[0]['center_x'] = 0.
    kwargs_source[0]['center_y'] = 0.

    jwst_ml = JWST(jwst_pix, exp_time, zp, m_sky, FWHM_150, psf_file, read_noise, ccd_gain,
                   psf_type="PIXEL", band="NIRCam_F150W", coadd_years=None)
    
    kwargs_cam_obs = jwst_ml.kwargs_single_band()

    sim = SimAPI(numpix = num_pix,
                 kwargs_single_band = kwargs_cam_obs,
                 kwargs_model = kwargs_model_cab)

    jwst_mock_class = sim.image_model_class(kwargs_numerics = {'point_source_supersampling_factor': 1})
    cab_lensed_img = jwst_mock_class.image(kwargs_lens=[kwargs_cab], kwargs_source=kwargs_source)

    sigma = pixel_errors[i]
    resid = (mock_imgs[i] - cab_lensed_img) / sigma

    return -0.5*np.sum((resid)**2.)

def ptform(u, minparam, maxparam):
    """Transforms samples `u` drawn from the unit cube to samples to those
    from our uniform prior within [-10., 10.) for each variable."""
    return minparam*(1.-u) + maxparam*u
        
##################################################################################
#                            sampling result analysis                            #
##################################################################################


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def bestfit(result):
    #takes in the result, returns best fit and errors
    #and returns -logl
    logs = result.logl
    samps = result.samples
    argmax = np.argmax(logs)

    weights = np.exp(result.logwt - result.logz[-1])
    mean, cov = dyfunc.mean_and_cov(samps, weights)

    errs = [cov[i,i] for i in range(len(mean))]

    return logs[argmax],samps[argmax],np.sqrt(errs)###*2.

def make_span_covering_truths(results, truths, frac=0.9995, pad_frac=0.2):
    """
    Returns a list of (lo, hi) bounds for each dim such that:
      1) it covers the equal-tailed `frac` of the posterior, and
      2) it expands to include the provided `truths`.
    """
    samps   = results.samples
    weights = np.exp(results.logwt - results.logz[-1])
    qlo, qhi = (1-frac)/2, 1-(1-frac)/2

    spans = []
    for j, t in enumerate(truths):
        lo, hi = dyfunc.quantile(samps[:, j], [qlo, qhi], weights=weights)
        width  = max(hi - lo, 1e-12)
        pad    = pad_frac * width
        if np.isfinite(t):
            if t < lo: lo = t - pad
            if t > hi: hi = t + pad
        spans.append((lo, hi))
    return spans

def make_dynesty_cornerplot(dresults, i, cab_model, npar, bestfs, all_cab_estimates, exp_time):

    #tfig2, axes = plt.subplots(N, N, figsize=(3 * N, 3 * N))

    labels_plot = ['rad_stretch',
                   'tan_stretch',
                   'curvature', 
                   'orientation']

    cab_estimates = all_cab_estimates[i]

    if cab_model == 'CURVED_ARC_SPT':
        labels_plot.extend(['gamma 1', 'gamma 2'])
        cab_estimates.extend([bestfs[4], bestfs[5]])

    if cab_model == 'CURVED_ARC_TAN_DIFF':
        labels_plot.append('dtan')
        cab_estimates.append(bestfs[4])

    quant = [0.6826894921370859, 0.9544997361036416]
    span  = make_span_covering_truths(dresults, cab_estimates, frac=0.9995, pad_frac=0.9)

    dyplot.cornerplot(dresults,
                      span=span,
                      color='royalblue',
                      labels=labels_plot,
                      truths=cab_estimates,
                      truth_color='red',
                      quantiles_2d=quant)

    plt.show()
    plt.close()
    
    
def render_cab_bestf_imgs(cab_model, kwargs_source, bestfs, z_source, z_lens, cosmo, num_pix, exp_time):

    kwargs_cab_bestf = {  'radial_stretch': bestfs[0],
                          'tangential_stretch': bestfs[1],
                          'curvature': bestfs[2],
                          'direction': bestfs[3],
                          'center_x': 0.,
                          'center_y': 0.,
                          }

    if cab_model == 'CURVED_ARC_SPT':
        kwargs_cab_bestf['gamma1'] = bestfs[4]
        kwargs_cab_bestf['gamma2'] = bestfs[5]

    if cab_model == 'CURVED_ARC_TAN_DIFF':
        kwargs_cab_bestf['dtan_dtan'] = bestfs[4]

    kwargs_model_cab = {'lens_model_list': [cab_model],
                        'z_lens': z_lens,
                        'lens_light_model_list': [],  
                        'source_light_model_list': ['INTERPOL'],  
                        'z_source': z_source,
                        'point_source_model_list': [],
                        'cosmo': cosmo}

    kwargs_source[0]['center_x'] = 0.
    kwargs_source[0]['center_y'] = 0.
    
    jwst_ml = JWST(jwst_pix, exp_time, zp, m_sky, FWHM_150, psf_file, read_noise, ccd_gain,
                   psf_type="PIXEL", band="NIRCam_F150W", coadd_years=None)

    kwargs_cam_obs = jwst_ml.kwargs_single_band()

    sim = SimAPI(numpix = num_pix,
                 kwargs_single_band = kwargs_cam_obs,
                 kwargs_model = kwargs_model_cab)

    jwst_mock_class = sim.image_model_class(kwargs_numerics = {'point_source_supersampling_factor': 1})
    cab_bestf_img = jwst_mock_class.image(kwargs_lens=[kwargs_cab_bestf], kwargs_source=kwargs_source)

    return cab_bestf_img


def render_cab_est_imgs(i, lens_ext, x_imgs, y_imgs, 
                        kwargs_source, kwargs_lens, z_lens, z_source, cosmo, 
                        all_cab_estimates, num_pix, exp_time):

    cab_estimates = []

    x_img, y_img = x_imgs[i], y_imgs[i]

    kwargs_cab = lens_ext.curved_arc_estimate(x_img, y_img, kwargs_lens)

    cab_estimates.append(kwargs_cab['radial_stretch'])
    cab_estimates.append(kwargs_cab['tangential_stretch'])
    cab_estimates.append(kwargs_cab['curvature'])
    cab_estimates.append(kwargs_cab['direction'])

    all_cab_estimates.append(cab_estimates)

    kwargs_cab['center_x'] = 0.
    kwargs_cab['center_y'] = 0.

    kwargs_model_cab = {'lens_model_list': ['CURVED_ARC_SIS_MST'], 
                        'z_lens': z_lens,
                        'lens_light_model_list': [],  
                        'source_light_model_list': ['INTERPOL'],  
                        'z_source': z_source,
                        'point_source_model_list': [],
                        'cosmo': cosmo}

    kwargs_source[0]['center_x'] = 0.
    kwargs_source[0]['center_y'] = 0.

    jwst_ml = JWST(jwst_pix, exp_time, zp, m_sky, FWHM_150, psf_file, read_noise, ccd_gain,
                   psf_type="PIXEL", band="NIRCam_F150W", coadd_years=None)

    kwargs_cam_obs = jwst_ml.kwargs_single_band()

    sim = SimAPI(numpix = num_pix,
                 kwargs_single_band = kwargs_cam_obs,
                 kwargs_model = kwargs_model_cab)

    jwst_mock_class = sim.image_model_class(kwargs_numerics = {'point_source_supersampling_factor': 1})
    cab_lensed_img = jwst_mock_class.image(kwargs_lens=[kwargs_cab], kwargs_source=kwargs_source)

    return cab_lensed_img, all_cab_estimates

################################################################################################
################################## Power Spectra Computation ###################################
################################################################################################

import imageio
import os
import scipy
import lenstronomy.Util.util as util
import matplotlib
from copy import deepcopy
import scipy.signal as signal

def get_pix_scales(num_pix=100, zoom_factor=1):

    deltaPix = jwst_pix / zoom_factor    
    rnge = num_pix * deltaPix
    
    return rnge

def make_mask(kx,ky,dk=1):
    x,y = np.meshgrid(ky,kx)
    k = np.sqrt(x**2+y**2)
    kmax = max(kx)
    dK = dk
    K = np.arange(kmax/dK)*dK
    mask_list =[]
    for i in range(len(K)):
        kmin = i*dK
        kmax = kmin + dK
        mask = (k >= kmin) * (k <= kmax)
        mask_list.append(mask*1)
    return mask_list

def twoD_ps(data,pix_size,rnge,shift=0,show_ps=False):
    """
    takes in a 2D array and returns the 2D FFT:
    inputs:
        data: n 2d arrays, whose average = coadd_coords
        pix_size: pixel size
        rnge: box size
        shift: offset from the halo center
        show_ps: whether you want to display the 2d power spectrum
    outputs:
        ind_ps: list of lists, where each list is a 2d power spectrum
        tot_ps: total 2D power spectrum after coadding all PS in ind_ps
        ky,ky: fft frequencies
    """

    A_pix = pix_size**2
    A_box = rnge**2
    
    ind_ps = []
    for i in data:
        ft = A_pix * np.fft.fft2(i)
        ps2D = np.abs(ft)**2 / A_box
        ind_ps.append(np.fft.fftshift(ps2D))

    tot_ps = np.mean(ind_ps,axis=0)

    kx = 2 * np.pi * np.fft.fftfreq(tot_ps.shape[0],d=pix_size)
    kx = np.fft.fftshift(kx)
    ky = 2 * np.pi * np.fft.fftfreq(tot_ps.shape[1],d=pix_size)
    ky = np.fft.fftshift(ky)

    if show_ps == True:
        tot_ps2 = np.log10(tot_ps)
        
        fig,(ax1,ax2) = plt.subplots(2,sharey=True)
        ax1.imshow(tot_ps,extent=[min(kx),max(kx),min(ky),max(ky)],interpolation='nearest')
        ax2.imshow(tot_ps2,extent=[min(kx),max(kx),min(ky),max(ky)],interpolation='nearest')
        plt.show()

    return ind_ps,tot_ps,kx,ky

def multipoles(data,num_pix,dr,x,y,mask=None,ns=[0]):
    """
        inputs:
        data: 2d map
        x,y: arrays that make up the map edges
        rnge: 2d map physical size
        dr: pixel size
	ns: a list of integers n where n represents the nth multipole (i.e. 0 = monopole, 1 = dipole, etc.)
    """
    
    data = np.asarray(data)
    shift = int(np.floor(num_pix/2))
    
    X,Y = np.meshgrid(x,y)
    r = np.sqrt(X**2+Y**2)

    if max(x) <= max(y):
        rmax = max(x)
    else:
        rmax = max(y)

    R = np.arange(rmax/dr)*dr

    power_spectra = {}

    for n in ns:
    	    
        power_spectra['%s' % n] = []
	
        if n == 0:

            for i,j in zip(range(len(R)),mask):
                ring = data*j
                pk = ring[ring != 0]
                dphi = 2*np.pi/len(pk)
                power_spectra['%s' % n].append((1/(2*np.pi))* np.sum(pk*dphi))
	
        if n != 0:    

            for i,j in zip(range(len(R)),mask):
                ring = data * j
        
                pk = ring[ring != 0]

                indices = np.asarray(np.nonzero(ring)) - shift
                dphi = 2 * np.pi / len(pk)

                phi = np.zeros(len(pk))
                count = 0
                for i,j in zip(indices[0],indices[1]):
                    phi[count] = np.arctan2(j,i)
                    count += 1
        
                integrand = pk * np.cos(n*phi)
	
                power_spectra['%s' % n].append((1/(2*np.pi)) * np.sum(dphi*integrand))

    return power_spectra,R

def variance(individual_ps,ps1d,N,kx,ky,rnge,num_pix,pix_size,mask=None,n=None):
    """
    takes in a list of lists where each nested list is a single 2d map and returns the variance of each map with respect to the average of all the maps
        individual_ps: list of lists where each nested list is a single 2d map
        ps1d: average 1D power spectrum
        kx,ky: fft frequencies
        rnge: spatial extent out to which we want to consider subhalos
        num_pix: number of pixels
        pix_size: pixel size
    """

    ns = [int(n)]

    var = []
    ind_curves = []
    for i in individual_ps:

        power_spectra,K = multipoles(i,num_pix,pix_size,kx,ky,mask=mask,ns=ns)
        diff = [(1/N)*(j-k)**2 for j,k in zip(power_spectra[n][1:],ps1d)]
        var.append(diff)
        ind_curves.append(power_spectra[n][1:])

    var = np.sum(var,axis=0)

    return var,ind_curves

import os

def compute_ps(data, name, cosmo, num_pix=100, zoom_factor=1):
               
    rnge = get_pix_scales(num_pix, zoom_factor)

    individual_ps,tot_ps,kx,ky = twoD_ps(data,jwst_pix,rnge)

    #load a mask and keep only the right number of rings (depends on pixel number)
        
    pix_size = jwst_pix

    X = np.linspace(-rnge/2.,rnge/2.,num_pix)
    Y = np.linspace(-rnge/2.,rnge/2.,num_pix)
    mask_list = make_mask(X,Y,dk=pix_size)

    orig_mask = mask_list
    
    bin_num = int((num_pix)/2 + 1)
    mask = []
    for i in orig_mask[:bin_num]:
        mask2 = []
        beg = int((len(i[0])-num_pix)/2)
        end = int(len(i[0])-beg)
        for j in i[beg:end]:
            mask2.append(j[beg:end])
        mask.append(mask2)

    pix_size_k = np.abs(kx[0]-kx[1])

    #obtain the 1d power spectrum for each convergence map
    ns = [0]
    power_spectra,K = multipoles(tot_ps,num_pix,pix_size_k,kx,ky,mask=mask,ns=ns)

    K = K[1:]

    dirps = os.path.join(
        '/n/netscratch', 'dvorkin_lab', 'Lab', 'nephremidze',
        'thesis', '0-cluster_substructure', 'power_spectra'
    )
    os.makedirs(dirps, exist_ok=True)
    
    # sanitize name for filenames
    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')

    # later, when you write the k file:
    dirk = os.path.join(
        dirps,
        f'k_{safe_name}_{num_pix:.0f}_{rnge:.2f}.txt'
    )
    with open(dirk, 'w') as file0:

        for i in K:
            # convert from arcsec to kpc
            arcsec_in_kpc = cosmo.arcsec2phys_lens(1.) * 1000
            i /= arcsec_in_kpc
            file0.write('%s\n' % i)
        file0.close()

    for key in power_spectra.keys():

        ps = power_spectra[key][1:] 
        _,ind_curves = variance(individual_ps,ps,len(individual_ps),kx,ky,rnge,num_pix,pix_size_k,mask=mask,n=key)

        dir2 = dirps + 'ind_curves_%s_%s_%s' % (name,round(num_pix),round(rnge, 2))
        np.save(dir2,ind_curves)
        
    return K, ps, ind_curves


# ################################################################################################
# ################################## Compute 2D WST Transforms ###################################
# ################################################################################################
# from kymatio.numpy import Scattering2D

# def compute_wst(batch_maps, name, J=2, L=8, max_order=2):

#     N, H, W = batch_maps.shape
    
#     scattering = Scattering2D(
#         J=J,
#         shape=(H, W),
#         L=L,
#         max_order=max_order,
#         pre_pad=False,
#         backend='numpy',
#         out_type='list'
#     )
    
#     Sx = scattering.scattering(batch_maps)
    
#     # Each dictionary in Sx contains keys:
#     #   'j': the scale(s) associated with this coefficient,
#     #   'theta': the orientation(s),
#     #   'coef': the scattering coefficient array with shape (N, downsampled_H, downsampled_W).
#     # Sort the coefficients by (j, theta) so that they are in a consistent order.
#     sorted_coeffs = sorted(
#         [(coeff['j'], coeff['theta'], coeff['coef']) for coeff in Sx],
#         key=lambda x: (x[0], x[1])
#     )
    
#     flattened_coeffs_list = []
#     for i in range(N):
#         sample_coeffs = []
#         for j_val, theta_val, coef in sorted_coeffs:
#             # For this scattering channel, coef[i] is a 2D array; flatten it to 1D.
#             sample_coeffs.append(coef[i].flatten())
#         # Concatenate all flattened arrays for sample i.
#         flattened_sample = np.concatenate(sample_coeffs)
#         flattened_coeffs_list.append(flattened_sample)
        
#     np.save(f'wst_coeffs/{name}_wst_coeffs_J={J}_L={L}_max_order_{max_order}.npy', flattened_coeffs_list)
    
#     return flattened_coeffs_list


##################################################################################
#                            lensing with substructure                           #
##################################################################################

class TruncationBoundMassPDFCustom(TruncationBoundMassPDF):
    """
    This class uses the latest tidal stripping model by Du & Gilman et al. (2025), but skips the calculation of the bound 
    mass based on the assumption that the deflector is a massive elliptical. 
    
    Here, we can specify the distribution of subhalo bound masses directly as a Gaussian in log10(mbound / m_infall). Then, we
    can use the tidal tracks to get the density profile. 

    Just set the values of fbound mean/sigma below, for this example the assumption is that cluster subhalos are stripped to 10% 
    of their infall masses, on average, with a standard deviation of 0.5 dex in log10(m_bound / m_infall). 
    """
    log10_fbound_mean = -1.0
    log10_fbound_standard_dev = 0.5
    def __init__(self, lens_cosmo):
        super(TruncationBoundMassPDFCustom, self).__init__(lens_cosmo, self.log10_fbound_mean, self.log10_fbound_standard_dev)

        
def make_subs_lens_model(num_pix, x_imgs, y_imgs, z_lens, z_source,
                        substructure_model, log10_dNdA, log_m_host, log_mhigh,
                        log10_m_uldm=None):
    
    los_angle = np.sqrt(2 * (num_pix * jwst_pix)**2)

    if substructure_model == 'NONE':

        lens_model_list, lens_redshift_list, kwargs_halos = [], [], [{}]

    if substructure_model == 'CDM':

        cluster_cdm = CDM(z_lens, z_source, 
                          log10_dNdA=log10_dNdA,
                          cone_opening_angle_arcsec=los_angle,
                          log_m_host=log_m_host, 
                          log_mhigh=log_mhigh,
                          truncation_model_subhalos=TruncationBoundMassPDFCustom,
                          infall_redshift_model='DIRECT_INFALL_CLUSTER',
                          geometry_type='DOUBLE_CONE')

        kappa_scale_subhalos = 10 ** -1.0 
                    # the average subhalo bound mass is now 10^log10_fbound_mean lower, so we rescale mass sheets accoringly
        kwargs_mass_sheet = {'kappa_scale_subhalos': kappa_scale_subhalos}
        lens_model_list, lens_redshift_list, kwargs_halos, _ = cluster_cdm.lensing_quantities(add_mass_sheet_correction=True,
                                                                                              kwargs_mass_sheet=kwargs_mass_sheet)
        
    if substructure_model == 'ULDM':
        
        n_cut = 50000 #600000 
                    #maximum number of fluctuations to render
        flucs_args = {'x_images': [0.], 'y_images': [0.], 'aperture': los_angle}
        
        cluster_uldm=ULDM(z_lens, z_source,
                          log10_m_uldm=log10_m_uldm,
                          uldm_plaw=1/3, flucs_shape='aperture', flucs_args=flucs_args,
                          log10_dNdA=log10_dNdA,
                          cone_opening_angle_arcsec=los_angle,
                          log_m_host=log_m_host, 
                          log_mhigh=log_mhigh,
                          truncation_model_subhalos=TruncationBoundMassPDFCustom,
                          geometry_type='DOUBLE_CONE',
                          n_cut=n_cut)

        kappa_scale_subhalos = 10 ** -1.0 
                    # the average subhalo bound mass is now 10^log10_fbound_mean lower, so we rescale mass sheets accoringly
        kwargs_mass_sheet = {'kappa_scale_subhalos': kappa_scale_subhalos}
        
        lens_model_list, lens_redshift_list, kwargs_halos, _ = cluster_uldm.lensing_quantities(add_mass_sheet_correction=True,
                                                                                               kwargs_mass_sheet=kwargs_mass_sheet)

    return lens_model_list, lens_redshift_list, kwargs_halos


def make_local_cluster_and_subs_lens(kwargs_lens_macro, cosmo,
                                     Rs_angle, alpha_Rs,
                                     num_pix, img_idx, x_imgs, y_imgs,
                                     z_lens, z_source, substructure_model,
                                     log10_dNdA, log_m_host, log_mhigh,
                                     log10_m_uldm=None):

    cluster_lens_model_list = ['TABULATED_DEFLECTIONS']

    cluster_local_model = ClusterLocalDeflection(Rs_angle = Rs_angle,
                                                 alpha_Rs = alpha_Rs,
                                                 num_pix = num_pix,
                                                 center_x = x_imgs[img_idx],
                                                 center_y = y_imgs[img_idx],
                                                 e1 = kwargs_lens_macro[0]['e1'],
                                                 e2 = kwargs_lens_macro[0]['e2'],
                                                 lens_center_x = kwargs_lens_macro[0]['center_x'],
                                                 lens_center_y = kwargs_lens_macro[0]['center_y'],
                                                 extrapolate=False)
    cluster_kwargs =[{}]

    (subs_lens_model_list,
     subs_lens_redshift_list, 
     subs_kwargs) = make_subs_lens_model(num_pix, x_imgs, y_imgs, z_lens, z_source,
                                         substructure_model, log10_dNdA, log_m_host, log_mhigh,
                                         log10_m_uldm=log10_m_uldm)

    lens_model_list_full = cluster_lens_model_list + subs_lens_model_list
    lens_z_list_full = [z_lens] + list(subs_lens_redshift_list)

    profile_kwargs_list_full = [{"custom_class": cluster_local_model}] + [{} for _ in subs_lens_model_list] 

    kwargs_lens_full = cluster_kwargs + subs_kwargs

    kwargs_model_full = {'lens_model_list': lens_model_list_full, 
                        'lens_profile_kwargs_list': profile_kwargs_list_full,
                        #'z_lens': z_lens,
                        'lens_redshift_list': lens_z_list_full,
                        'lens_light_model_list': [],  
                        'source_light_model_list': ['INTERPOL'],  
                        'z_source': z_source,
                        'point_source_model_list': [],
                        'cosmo': cosmo}

    return kwargs_lens_full, kwargs_model_full

    
def generate_mock_set(cosmo, kwargs_lens_macro, kwargs_source,
                      Rs_angle, alpha_Rs,
                      x_imgs, y_imgs, z_lens, z_source,
                      substructure_model,
                      log10_dNdA, log_m_host, log_mhigh,
                      mock_set_idx, base_seed, exp_time, num_pix, out_dir,
                      parallel_cores = 0, # no parallel processing by default
                      save_mocks = True,
                      plot_figs = True,
                      save_figs = True,
                      log10_m_uldm = None):
    
    lensed_imgs, noises, mock_imgs, pixel_errors = [], [], [], []
    all_models_list, all_lenses_list = [], []
    
    for img_idx in range(len(x_imgs)):
    
        kwargs_lens_full, kwargs_model_full = make_local_cluster_and_subs_lens(kwargs_lens_macro, cosmo,
                                                                         Rs_angle, alpha_Rs,
                                                                         num_pix, img_idx, x_imgs, y_imgs,
                                                                         z_lens, z_source, substructure_model,
                                                                         log10_dNdA, log_m_host, log_mhigh,
                                                                         log10_m_uldm=log10_m_uldm)
        
        (lensed_img, noise, 
         jwst_mock, sigma_fixed) = generate_ith_mocks(num_pix,
                                                      kwargs_source, kwargs_lens_full, kwargs_model_full,
                                                      z_source, cosmo, exp_time,
                                                      parallel_cores = parallel_cores)
        
        lensed_imgs.append(lensed_img)
        noises.append(noise)
        mock_imgs.append(jwst_mock)
        pixel_errors.append(sigma_fixed)
        
        all_models_list.append(kwargs_model_full)
        all_lenses_list.append(kwargs_lens_full)

    if save_mocks:

        np.save(f"{out_dir}/subs={substructure_model}_set_#{mock_set_idx+1}_seed={base_seed}_mock_imgs.npy", mock_imgs)
        np.save(f"{out_dir}/subs={substructure_model}_set_#{mock_set_idx+1}_seed={base_seed}_mock_errs.npy", pixel_errors)

    if plot_figs:
        
        visualize_mock_set(substructure_model, lensed_imgs, noises, mock_imgs, save_figs, exp_time, out_dir, mock_set_idx)
        
        make_magnifications_plot(mock_set_idx, base_seed, num_pix, x_imgs, out_dir,
                                 all_models_list, all_lenses_list, substructure_model,
                                 parallel_cores = parallel_cores,
                                 save_figs = save_figs)
        
    return lensed_imgs, noises, mock_imgs, pixel_errors, all_models_list, all_lenses_list

##################################################################################
#                        Parallelizing lensing computations                      #
##################################################################################

import os
import multiprocessing as mp
import numpy as np
from scipy.signal import fftconvolve

def _tile_tasks(num_pix: int, tile_size: int):
    """
    Return a list of square tiles (x0, x1, y0, y1) that exactly tessellate the image.
    Requires num_pix to be divisible by tile_size so tiles are all tile_size x tile_size.
    """
    if num_pix % tile_size != 0:
        raise ValueError(f"num_pix ({num_pix}) must be divisible by tile_size ({tile_size}) "
                         "to render strict tile_size x tile_size subgrids.")
    edges = list(range(0, num_pix + 1, tile_size))
    tiles = []
    for yi in range(len(edges) - 1):
        for xi in range(len(edges) - 1):
            x0, x1 = edges[xi], edges[xi + 1]
            y0, y1 = edges[yi], edges[yi + 1]
            tiles.append((x0, x1, y0, y1))
    return tiles

# ---- WCS shifter (keeps local (0,0) convention) --------------------
def _shift_kwargs_data_for_tile(kwargs_data_full: dict, x0: int, y0: int) -> dict:
    """
    Return a copy of kwargs_data_full with ra_at_xy_0, dec_at_xy_0 shifted so that
    the tile’s pixel (0,0) corresponds to the global pixel (x0, y0).
    """
    kd = dict(kwargs_data_full)
    kd = {k: v for k, v in kd.items() if k not in ("kernel_point_source", "kernel_pixel", "kwargs_psf")}
    kd["psf_type"] = "NONE"

    kpg = dict(kd["kwargs_pixel_grid"])
    A = np.array(kpg["transform_pix2angle"], dtype=float)
    shift_ra, shift_dec = A @ np.array([x0, y0], dtype=float)
    kpg["ra_at_xy_0"] += shift_ra
    kpg["dec_at_xy_0"] += shift_dec

    kd["kwargs_pixel_grid"] = kpg
    return kd

# ---- worker ---------------------------------------------------------
def _render_tile_simapi(task):
    """
    Worker: render one tile (without PSF) using SimAPI with a WCS shifted so tile
    coordinates remain in the same local frame as the full image.
    """
    from lenstronomy.SimulationAPI.sim_api import SimAPI
    from lenstronomy.ImSim.image_model import ImageModel

    (kw_data_no_psf_base, kwargs_model_full, kwargs_lens_full, kwargs_source,
     tile_size) = task["globals"]
    x0, x1, y0, y1 = task["tile"]

    # Build per-tile kwargs_single_band and size (square tile)
    kd = _shift_kwargs_data_for_tile(kw_data_no_psf_base, x0, y0)
    tile_npix = x1 - x0  # == y1 - y0 by construction

    # SimAPI builds the Data/PSF/Lens/Source classes consistently
    sim = SimAPI(numpix=tile_npix,
                 kwargs_single_band=kd,
                 kwargs_model=kwargs_model_full)
    data_class = sim.data_class
    psf_class = sim.psf_class     # PSF is NONE here
    lens_model_class = sim.lens_model_class
    source_model_class = sim.source_model_class

    # ImageModel to ray-trace this tile
    image_model = ImageModel(data_class, psf_class, lens_model_class,
                             source_model_class=source_model_class)

    # Ray-trace (no PSF)
    sub = image_model.image(kwargs_lens_full, kwargs_source)  # shape (tile_npix, tile_npix)
    return (x0, x1, y0, y1, sub)

# ---- main entry -----------------------------------------------------
def image_parallel_tiles_then_psf(
    *,
    num_pix: int,
    jwst_pix: float,                 # (unused here but kept for signature stability)
    kwargs_data_full: dict,          # from JWST(...).kwargs_single_band(), centered so image center -> (0,0)
    kwargs_model: dict,              # == kwargs_model_full (cluster TABULATED_DEFLECTIONS + subhalos)
    kwargs_lens: list,               # == kwargs_lens_full
    kwargs_source: list,             # your source kwargs
    psf_kernel: np.ndarray,
    tile_size: int = 10,
    processes: int = 112
):
    """
    Parallel rendering by assigning tile_size x tile_size tiles to workers,
    ray-tracing each WITHOUT PSF, then a single PSF convolution at the end.

    Compatibility expectations:
    - kwargs_model['lens_model_list'] includes 'TABULATED_DEFLECTIONS' whose
      profile_kwargs_list has per-image custom_class=ClusterLocalDeflection(...)
      with (center_x, center_y) baked in.
    - kwargs_lens for that entry should be {} (no center_x/center_y there).
    - kwargs_data_full encodes a WCS centered so that the full image center maps to (0,0).
    """
    # 1) Plan tiles (strict squares)
    tiles = _tile_tasks(num_pix, tile_size)

    # 2) Base kwargs_single_band with PSF disabled for workers
    kw_data_no_psf_base = dict(kwargs_data_full)
    kw_data_no_psf_base["psf_type"] = "NONE"
    kw_data_no_psf_base.pop("kernel_point_source", None)
    kw_data_no_psf_base.pop("kernel_pixel", None)
    kw_data_no_psf_base.pop("kwargs_psf", None)

    # 3) Bundle globals once to reduce pickling cost
    g = {
        "globals": (kw_data_no_psf_base, kwargs_model, kwargs_lens, kwargs_source, tile_size)
    }
    task_args = [dict(g, tile=t) for t in tiles]

    # 4) Parallel render
    ctx = mp.get_context("spawn")  # robust across platforms
    procs = min(processes, os.cpu_count() or 1)
    with ctx.Pool(processes=procs) as pool:
        parts = pool.map(_render_tile_simapi, task_args, chunksize=1)

    # 5) Assemble raw (unconvolved) mosaic
    raw = np.zeros((num_pix, num_pix), dtype=float)
    for (x0, x1, y0, y1, sub) in parts:
        raw[y0:y1, x0:x1] = sub  # sub is tile_size x tile_size

    # 6) Single PSF convolution at the very end
    k = psf_kernel / np.sum(psf_kernel)
    final = fftconvolve(raw, k, mode="same")
    return final, raw


##################################################################################
#                   Illustrative local magnifications plot                      #
##################################################################################

import os
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Util import util as _util

def make_kwargs_data_for_mu(num_pix: int, jwst_pix: float) -> dict:
    """Minimal kwargs_data_full (only kwargs_pixel_grid) centered at (0,0)."""
    xc = yc = (num_pix - 1) / 2.0
    (_, _, _ra0, _dec0, _, _, M, _) = _util.make_grid_with_coordtransform(
        numPix=num_pix, deltapix=jwst_pix, subgrid_res=1, left_lower=False, inverse=False
    )
    ra_at_xy_0  = - (M[0, 0]*xc + M[0, 1]*yc)
    dec_at_xy_0 = - (M[1, 0]*xc + M[1, 1]*yc)
    return {"kwargs_pixel_grid": dict(ra_at_xy_0=ra_at_xy_0, dec_at_xy_0=dec_at_xy_0, transform_pix2angle=M)}

def _build_angle_grid_from_kwargs(kwargs_data_full: dict, num_pix: int):
    """Angle grid matching your mocks (arcsec)."""
    kpg = kwargs_data_full["kwargs_pixel_grid"]
    A   = np.array(kpg["transform_pix2angle"], dtype=float)
    ra0 = float(kpg["ra_at_xy_0"]); dec0 = float(kpg["dec_at_xy_0"])
    j = np.arange(num_pix, dtype=float); i = np.arange(num_pix, dtype=float)
    jj, ii = np.meshgrid(j, i)
    xx = ra0 + A[0, 0]*jj + A[0, 1]*ii
    yy = dec0 + A[1, 0]*jj + A[1, 1]*ii
    return xx, yy

def _lensmodel_from_kwargs(kwargs_model: dict) -> LensModel:
    """Construct a LensModel from kwargs_model_full."""
    lm_kwargs = {"lens_model_list": kwargs_model["lens_model_list"]}
    if "lens_profile_kwargs_list" in kwargs_model:
        lm_kwargs["profile_kwargs_list"] = kwargs_model["lens_profile_kwargs_list"]
    if "lens_redshift_list" in kwargs_model and "z_source" in kwargs_model:
        lm_kwargs["lens_redshift_list"] = kwargs_model["lens_redshift_list"]
        lm_kwargs["z_source"] = kwargs_model["z_source"]
        if "cosmo" in kwargs_model:
            lm_kwargs["cosmo"] = kwargs_model["cosmo"]
        lm_kwargs["multi_plane"] = True
    return LensModel(**lm_kwargs)

def _magnification_tile_worker(task):
    kwargs_model, kwargs_lens, xx, yy, tile = task
    x0, x1, y0, y1 = tile
    lens_model = _lensmodel_from_kwargs(kwargs_model)
    mu = lens_model.magnification(xx[y0:y1, x0:x1], yy[y0:y1, x0:x1], kwargs_lens)
    return (x0, x1, y0, y1, mu)

def parallel_magnification_total(num_pix: int,
                                 kwargs_data_full: dict,
                                 kwargs_model: dict,
                                 kwargs_lens: list,
                                 *,
                                 tile_size: int = 10,
                                 processes: int = 100):
    """Total magnification μ on the mock grid, parallel over tiles. Returns (mu, xx, yy)."""
    xx, yy = _build_angle_grid_from_kwargs(kwargs_data_full, num_pix)
    tiles  = _tile_tasks(num_pix, tile_size)
    tasks  = [(kwargs_model, kwargs_lens, xx, yy, t) for t in tiles]

    mu = np.zeros((num_pix, num_pix), dtype=float)

    if processes == 0:
        # Serial path: no Pool, no spawn overhead
        for task in tasks:
            x0, x1, y0, y1, sub_mu = _magnification_tile_worker(task)
            mu[y0:y1, x0:x1] = sub_mu
    else:

        import os, multiprocessing as mp
        procs = min(processes, os.cpu_count() or 1)
    
        ctx = mp.get_context("spawn")  # or "fork" on Linux if appropriate
        with ctx.Pool(procs) as pool:
            for (x0, x1, y0, y1, sub_mu) in pool.imap_unordered(_magnification_tile_worker, tasks, chunksize=1):
                mu[y0:y1, x0:x1] = sub_mu

    return mu, xx, yy

def plot_mock_set_mags(mus, *,
                         titles=None,
                         use_abs=True,
                         cmap="viridis",
                         figsize=(15, 5)):

    if titles is None:
        titles = [f"Image {i+1}" for i in range(len(mus))]

    fig, axes = plt.subplots(1, len(mus), figsize=figsize, constrained_layout=True)

    for ax, mu, ttl in zip(axes, mus, titles):
        img = np.abs(mu) if use_abs else mu
        finite = img[np.isfinite(img)]
        if finite.size:
            vmin = np.nanpercentile(finite, 1)
            vmax = np.nanpercentile(finite, 99)
        else:
            vmin, vmax = 0.0, 1.0

        im = ax.imshow(img, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(ttl)
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("magnification")

    return fig, axes


##################################################################################
#                              Make dynesty cornerplot                           #
##################################################################################


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def bestfit_mean(result):
    #takes in the result, returns best fit and errors
    #and returns -logl
    logs = result.logl
    samps = result.samples
    argmax = np.argmax(logs)

    weights = np.exp(result.logwt - result.logz[-1])
    mean, cov = dyfunc.mean_and_cov(samps, weights)

    errs = [cov[i,i] for i in range(len(mean))]

    return logs[argmax],mean,np.sqrt(errs)###*2.

def bestfit(result):
    #takes in the result, returns best fit and errors
    #and returns -logl
    logs = result.logl
    samps = result.samples
    argmax = np.argmax(logs)

    weights = np.exp(result.logwt - result.logz[-1])
    mean, cov = dyfunc.mean_and_cov(samps, weights)

    errs = [cov[i,i] for i in range(len(mean))]

    return logs[argmax],samps[argmax],np.sqrt(errs)###*2. 

def bic(logl, cab_model, nmax, num_pix):

    ndeg = num_pix**2

    if cab_model == 'CURVED_ARC_SIS_MST':
        N = 3*4 - 1 + 4 + 3

    elif cab_model == 'CURVED_ARC_SPT':
        N = 3*6 - 1 + 4 + 3

    elif cab_model == 'CURVED_ARC_TAN_DIFF':
        N = 3*5 - 1 + 4 + 3

    kparam = N + (nmax+1)*(nmax+2)/2
    return kparam*np.log(ndeg) - 2.*logl

def overplot_models(datasets, cab_truths, cab_model, num_pix, std=5., y_height=2., nmax=20, all_plot_limits=None, all_y_lims=None):
    
    mpl.rcdefaults()

    lbl_font = 35
    tck_font = 30
    lgnd_font = 40

    font = {
        'family': 'serif',
        'weight': 'normal',
        'size': tck_font}
    
    matplotlib.rc('font', **font)

    mpl.rcParams['xtick.major.pad'] = 5
    mpl.rcParams['ytick.major.pad'] = 5

    mpl.rcParams['xtick.major.width'] = 2.0
    mpl.rcParams['ytick.major.width'] = 2.0
    mpl.rcParams['xtick.minor.width'] = 1.5
    mpl.rcParams['ytick.minor.width'] = 1.5

    mpl.rcParams['xtick.major.size']  = 8
    mpl.rcParams['ytick.major.size']  = 8
    mpl.rcParams['xtick.minor.size']  = 4
    mpl.rcParams['ytick.minor.size']  = 4

    truedic = {'linewidth': 2., 'linestyle': 'dashed'}
    histdic = {'density': True, 'alpha': 0.5}

    filters = [115, 150, 200]

    colors = ['darkviolet', 'darkslateblue', 'darkolivegreen']

    labls_mask = [
        r'$\lambda_{\mathrm{tan},1}$',
        r'$s_{\mathrm{tan},1}$ ["$^{-1}$]',   # arcsec^{-1}
        r'$\phi_{1}/\pi$',
        r'$\lambda_{\mathrm{rad},2}$',
        r'$\lambda_{\mathrm{tan},2}$',
        r'$s_{\mathrm{tan},2}$ ["$^{-1}$]',
        r'$\phi_{2}/\pi$',
        # r'$\alpha_{2,x}$ ["]',             # arcsec
        # r'$\alpha_{2,y}$ ["]',
        r'$\lambda_{\mathrm{rad},3}$',
        r'$\lambda_{\mathrm{tan}}$',
        r'$s_{\mathrm{tan},3}$ ["$^{-1}$]',
        r'$\phi_{3}/\pi$',
        # r'$\alpha_{3,x}$ ["]',
        # r'$\alpha_{3,y}$ ["]',
        r'$\delta_1$ ["]',
        r'$x_1$ ["]',
        r'$y_1$ ["]']

    N = 11

    all_samples = []
    results = []
    all_bestfits = []
    all_errors = []

    all_upper_bounds = []
    all_lower_bounds = []

    for model_idx in range(len(datasets)):

        path = datasets[model_idx]
        dresults = load_obj(path)

        #plt.figure()
        # dyplot.cornerplot(dresults,
        #                   max_n_ticks=1,
        #                   color=colors[model_idx])

        if model_idx == 0:
            nlist = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12]
        elif model_idx == 1: #'CURVED_ARC_SPT':
            nlist = [0, 1, 2, 5, 6, 7, 8, 13, 14, 15, 16]
        elif model_idx == 2: #'CURVED_ARC_TAN_DIFF':
            nlist = [0, 1, 2, 4, 5, 6, 7, 11, 12, 13, 14]

        posterior_samples = dresults['samples'][:, nlist]
        all_samples.append(posterior_samples)
        results.append(dresults)
        maxl, bestf, cv = bestfit(dresults)

        bestfpp = bestf[nlist]
        errors = cv[nlist]

        upper_bounds = bestfpp + std * errors
        lower_bounds = bestfpp - std * errors

        all_bestfits.append(bestfpp)
        all_errors.append(errors)

        all_upper_bounds.append(upper_bounds)
        all_lower_bounds.append(lower_bounds)

    # --- build model-driven bounds as you already do ---
    abs_upper_bounds = np.max(all_upper_bounds, axis=0)
    abs_lower_bounds = np.min(all_lower_bounds, axis=0)

    # --- ensure truths are inside the bounds (with a small pad) ---
    truths_arr = np.asarray(cab_truths, dtype=float)

    # If cab_truths is longer than the plotted N, crop; if shorter, pad with NaN
    if truths_arr.shape[0] < N:
        truths_arr = np.pad(truths_arr, (0, N - truths_arr.shape[0]), constant_values=np.nan)
    else:
        truths_arr = truths_arr[:N]

    # Only act on finite truths
    finite = np.isfinite(truths_arr)

    # First, force bounds to at least include the truth value
    abs_lower_bounds[finite] = np.minimum(abs_lower_bounds[finite], truths_arr[finite])
    abs_upper_bounds[finite] = np.maximum(abs_upper_bounds[finite], truths_arr[finite])

    # Add a small margin if the truth touches or is extremely close to the boundary
    rng = abs_upper_bounds - abs_lower_bounds
    pad = np.maximum(1e-12, 0.05 * rng)  # 5% of current range (>= 1e-12)

    touch_low  = finite & (truths_arr <= abs_lower_bounds + 1e-12)
    touch_high = finite & (truths_arr >= abs_upper_bounds - 1e-12)

    abs_lower_bounds[touch_low]  = truths_arr[touch_low]  - pad[touch_low]
    abs_upper_bounds[touch_high] = truths_arr[touch_high] + pad[touch_high]

    # If user provided all_plot_limits, still guarantee truths are inside those too
    if all_plot_limits is not None:
        abs_upper_bounds = np.asarray(all_plot_limits[0], dtype=float)
        abs_lower_bounds = np.asarray(all_plot_limits[1], dtype=float)

        # expand user limits to include truths + pad
        rng = np.maximum(1e-12, abs_upper_bounds - abs_lower_bounds)
        pad = np.maximum(0.5 * rng)

        need_low_expand  = finite & (truths_arr < abs_lower_bounds)
        need_high_expand = finite & (truths_arr > abs_upper_bounds)

        abs_lower_bounds[need_low_expand]  = truths_arr[need_low_expand]  - pad[need_low_expand]
        abs_upper_bounds[need_high_expand] = truths_arr[need_high_expand] + pad[need_high_expand]

    # finally, build span tuples
    span = [(abs_lower_bounds[i], abs_upper_bounds[i]) for i in range(N)]

    max_y_values = [0] * N

    if all_plot_limits != None:

        abs_upper_bounds = all_plot_limits[0]
        abs_lower_bounds = all_plot_limits[1]

        span = [(abs_lower_bounds[i], abs_upper_bounds[i]) for i in range(N)]

    bestfits = []
    errors = []
    all_means = []   

    tfig2, axes = plt.subplots(N, N, figsize=(3 * N, 3 * N))

    for model_idx in range(len(datasets)):

        dresults = results[model_idx]

        if model_idx == 0:
            nlist = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12]
        elif model_idx == 1: #'CURVED_ARC_SPT':
            nlist = [0, 1, 2, 5, 6, 7, 8, 13, 14, 15, 16]
        elif model_idx == 2: #'CURVED_ARC_TAN_DIFF':
            nlist = [0, 1, 2, 4, 5, 6, 7, 11, 12, 13, 14]

        posterior_samples = dresults['samples'][:, nlist]
        maxl, bestf, error = bestfit(dresults)  
        maxl, means, error = bestfit_mean(dresults)
        bestfits.append(bestf)
        errors.append(error)
        all_means.append(means)

        bestfpp = bestf[nlist]
        err_pp = error[nlist]
        means_pp = means[nlist]

        quant = [0.6826894921370859, 0.9544997361036416]

        # calculate BIC
        BICs = []
        bic_value = bic(maxl, cab_model, nmax, num_pix)

        BICs.append(bic_value)

        # evidence
        evidences = []
        log_z_value = dresults.logz[-1]
        evidences.append(log_z_value)


        labels_plot = [labls_mask[i] for i in range(len(nlist))]

        tfig2, axes = dyplot.cornerplot(
            dresults,
            span=span,
            dims=nlist,
            quantiles=[],
            quantiles_2d=quant,
            show_titles=False,
            truths=cab_truths,
            max_n_ticks=3,
            truth_color='red',
            labels=labels_plot,
            color=colors[model_idx],
            fig=[tfig2, axes],
            hist_kwargs=histdic,
            truth_kwargs=truedic,
            label_kwargs={'fontsize': lbl_font}
        )

        for i in range(N):
            param_samples = posterior_samples[:, i]
            bins = 100 
            hist, bin_edges = np.histogram(param_samples, bins=bins, range=span[i], density=True)
            max_hist = np.max(hist)
            if max_hist > max_y_values[i]:
                max_y_values[i] = max_hist

    for i in range(N):
        ax = axes[i, i]
        upper_lim = max_y_values[i] * y_height 

        if all_y_lims != None:
            upper_lim = all_y_lims[i]

        ax.set_ylim(0, upper_lim)

    color_map = {115: colors[0], 150: colors[1], 200: colors[2]}

    handles = [
        Patch(color=color_map[115], label='CAB', alpha=0.5),
        Patch(color=color_map[150], label='CAB+shear', alpha=0.5),
        Patch(color=color_map[200], label='CAB+dtan', alpha=0.5),
        Line2D([0], [0], color='red', linestyle='--', label='CAB estimate')
    ]

    tfig2.subplots_adjust(left=0.06, bottom=0.06)

    for ax in axes[-1, :]:
        ax.xaxis.set_label_coords(0.5, -0.5)
    for ax in axes[:, 0]:
        ax.yaxis.set_label_coords(-0.5, 0.5)

    tfig2.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=lgnd_font)


##################################################################################
#                         Plot best-fit model reconstructions                    #
##################################################################################

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

def model_curved_fit_shapelet_sers(data,kappashear_params,source_params,likemask_list,indices,kwargs_numerics,data_class2,psf_class):   
    A_list = []
    C_D_response_list = []
    d_list = []
    for i in indices:
    # the lens model is a supperposition of an elliptical lens model with external shear
        lens_model_list_new = ['SHIFT','CURVED_ARC_SIS_MST']
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

def plot_true_source(source_kwargs,param,exp_time,sigma_bkg,kwargs_numerics):
    # data specifics
    numPix = 720  #  cutout pixel size
    deltaPix = 0.002  #  pixel size in arcsec (area per pixel = deltaPix**2)
    fwhm = 4e-2  # full width half max of PSF (only valid when psf_type='gaussian')
    psf_type = 'GAUSSIAN'  # 'GAUSSIAN', 'PIXEL', 'NONE'
    # generate the psf variables
    kernel_cut = np.zeros([15,15])
    kernel_cut[7,7] = 1.
    kwargs_psf = {'psf_type': psf_type, 'pixel_size': deltaPix, 'fwhm': fwhm, 'kernel_point_source':kernel_cut}
    #kwargs_psf = sim_util.psf_configure_simple(psf_type=psf_type, fwhm=fwhm, kernelsize=kernel_size, deltaPix=deltaPix, kernel=kernel)
    psf_class = PSF(**kwargs_psf)
    
    # generate the coordinate grid and image properties
    kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, sigma_bkg)
    kwargs_data['exposure_time'] = exp_time * np.ones_like(kwargs_data['image_data'])
    data_class = ImageData(**kwargs_data)
    # keywords for PixelGrid() class used in plotting routines
    kwargs_pixel_grid = {'nx': numPix, 'ny': numPix, 'transform_pix2angle': kwargs_data['transform_pix2angle'],
                         'ra_at_xy_0': kwargs_data['ra_at_xy_0'], 'dec_at_xy_0': kwargs_data['dec_at_xy_0']}
 
    # the lens model is a supperposition of an elliptical lens model with external shear
    lens_model_list_new = []
    kwargs_lens_true_new = []

    lens_model_class = LensModel(lens_model_list=lens_model_list_new)

    source_model_list = ['SHAPELETS']
    source_model_class = LightModel(light_model_list=source_model_list)
    
    source_params0 = copy.deepcopy(source_kwargs[0])
    source_params0['amp'] = param
        
    imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class, kwargs_numerics=kwargs_numerics)

    # generate image
    image_sim = imageModel.image(kwargs_lens_true_new, [source_params0])
        
    return image_sim

def visualize(params, nmax, imagearr, likearr, indices,
             kwargs_numerics, data_class2, psf_class,
             flatarr, flaterror, exp_time, sigma_bkg):    
    
    mur1, mut1 = 1.,params[0]
    mur2, mut2 = params[3],params[4]
    mur3, mut3 = params[9],params[10]
    
    cv1 = params[1] #np.abs(params[1])
    cv2 = params[5] #np.abs(params[5])
    cv3 = params[11] #np.abs(params[11])

    psi_ext1 = params[2]#*np.pi - (np.sign(params[1])+1.)*np.pi/2.
    psi_ext2 = params[6]#*np.pi - (np.sign(params[5])+1.)*np.pi/2.
    psi_ext3 = params[12]#*np.pi - (np.sign(params[11])+1.)*np.pi/2.

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

    fit,paramq = model_curved_fit_shapelet_sers(imagearr,kwargs_kapshe,source_shape,likearr,indices,kwargs_numerics,data_class2,psf_class)
    
    lnlout = -0.5*np.sum((((fit-flatarr)/flaterror)**2.))
    
    sourcefit = plot_true_source(source_shape,paramq,exp_time,sigma_bkg,kwargs_numerics)
    unflatfit = unflatten(fit,likearr)
    return  unflatfit,sourcefit,lnlout


def visualize_bestf_imgs(sourcevis, imagevis, 
                         array200sci, array200sci2, array200sci3, 
                         arytsterr, arytsterr2, arytsterr3, 
                         likemask, likemask2, likemask3,
                         plot_figs=True):
    
    res1 = (array200sci-imagevis[0])*likemask/arytsterr
    res2 = (array200sci2-imagevis[1])*likemask2/arytsterr2
    res3 = (array200sci3-imagevis[2])*likemask3/arytsterr3
    
    if plot_figs:

        cmap_string = 'jet'
        cmap = plt.get_cmap(cmap_string)
        cmap.set_bad(color='k', alpha=1.)
        cmap.set_under('k')

        scal = 3.

        f, axes = plt.subplots(4, 3, figsize=(15*scal,20*scal))
        ax = axes

        fontsss = 32

        txtclr = 'Black'

        vmin = 1e-6
        vmax = np.nanmax(array200sci)

        ax[0,0].set_title('#1',fontsize = fontsss*scal, color=txtclr)
        ax[0,1].set_title('#2',fontsize = fontsss*scal, color=txtclr)
        ax[0,2].set_title('#3',fontsize = fontsss*scal, color=txtclr)

        ax[0,0].set_ylabel('Images',fontsize = fontsss*scal, color=txtclr)
        ax[1,0].set_ylabel('Reconstructions',fontsize = fontsss*scal, color=txtclr)
        ax[2,0].set_ylabel('Residuals',fontsize = fontsss*scal, color=txtclr)

        im0 = ax[0,0].imshow(array200sci*likemask, origin='upper', vmin=vmin, vmax=vmax, cmap=cmap)
        im0 = ax[0,1].imshow(array200sci2*likemask2, origin='upper', vmin=vmin, vmax=vmax, cmap=cmap)
        im0 = ax[0,2].imshow(array200sci3*likemask3, origin='upper', vmin=vmin, vmax=vmax, cmap=cmap)

        x1 = [5,5+1/jwst_pix]
        y1 = [115,115]

        im0 = ax[1,0].imshow(imagevis[0]*likemask, origin='upper', vmin=vmin, vmax=vmax, cmap=cmap)
        im0 = ax[1,1].imshow(imagevis[1]*likemask2, origin='upper', vmin=vmin, vmax=vmax, cmap=cmap)
        im0 = ax[1,2].imshow(imagevis[2]*likemask3, origin='upper', vmin=vmin, vmax=vmax, cmap=cmap)

        vmn = -5.
        vmx = 5.

        cmap2 = 'bwr'

        im0 = ax[2,0].imshow(res1, origin='upper', vmin=vmn, vmax=vmx, cmap=cmap2)
        im0 = ax[2,1].imshow(res2, origin='upper', vmin=vmn, vmax=vmx, cmap=cmap2)
        im0 = ax[2,2].imshow(res3, origin='upper', vmin=vmn, vmax=vmx, cmap=cmap2)

        for q in range(0,3):
            for axk in ax[q,:]:
                axk.set_xticklabels([])
                axk.set_yticklabels([])

                axk.set_xticks([])
                axk.set_yticks([])

                for direc in ['left', 'right', 'top', 'bottom']:

                    axk.spines[direc].set_color('red') 
                    axk.spines[direc].set_linewidth(3*scal)

        f.add_artist(lines.Line2D([-0.01,1.01], [0.5,0.5],alpha=0.))

        vmins = np.log10(np.nanmax(sourcevis))-1.
        vmaxs = np.log10(np.nanmax(sourcevis))

        ax = axes
        gs = axes[0,0].get_gridspec()
        axbig = axes[3,0]

        im0 = axbig.imshow(np.log10(sourcevis), origin='upper', vmin=vmins, vmax=vmaxs, cmap=cmap)

        axbig.set_xticks([])
        axbig.set_yticks([])

        axbig.set_xticklabels([])
        axbig.set_yticklabels([])

        for direc in ['left', 'right', 'top', 'bottom']:

            axbig.spines[direc].set_color('red') 
            axbig.spines[direc].set_linewidth(3*scal)

        axbig.set_ylabel('Source Model',fontsize = fontsss*scal, color=txtclr)

        deltaPix_s = 0.002

        scaleg = 6

        x1s = [5*scaleg,5*scaleg+0.2*2/deltaPix_s]
        y1s = [115*scaleg,115*scaleg]
        axbig.add_artist(lines.Line2D(x1s, y1s,color="white", linewidth=3*scal))
        axbig.text(5*scaleg,112*scaleg, '0.2"', fontsize = fontsss*scal,color='White')

        axes[3,1].remove()
        axes[3,2].remove()

        plt.tight_layout()
    
    return res1, res2, res3