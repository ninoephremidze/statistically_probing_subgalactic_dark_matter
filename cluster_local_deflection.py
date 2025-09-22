import numpy as np
from scipy.interpolate import RegularGridInterpolator
from lenstronomy.LensModel.lens_model import LensModel

class ClusterLocalDeflection:
    """
    Precompute (alpha_x, alpha_y) for an NFW_ELLIPSE_CSE lens on a grid that is
    centered at (center_x, center_y) and whose side length is num_pix * jwst_pix.
    The stored/interpolated field is *local*: alpha(x,y) - alpha(center_x,center_y).
    Calling with local offsets (x, y) returns that local lensing field evaluated at
    (center_x + x, center_y + y), so (0, 0) maps to (0, 0).
    """

    def __init__(self,
                 Rs_angle,
                 alpha_Rs,
                 num_pix,
                 center_x=0.0,
                 center_y=0.0,
                 e1=0.27,
                 e2=0.0,
                 lens_center_x=0.0,
                 lens_center_y=0.0,
                 extrapolate=False,
                 *,
                 jwst_pix = 0.031230659851709842):
        """
        Args
        ----
        Rs_angle, alpha_Rs : NFW_ELLIPSE_CSE parameters (lenstronomy units)
        num_pix            : number of grid samples per axis (also sets side length)
        center_x, center_y : center of the interpolation grid [arcsec]
        e1, e2             : ellipticity components
        lens_center_x/y    : analytic lens center used to generate the grid
        extrapolate        : allow smooth extrapolation outside the grid
        jwst_pix           : pixel scale [arcsec]; kept fixed and multiplied by num_pix
        """

        if num_pix is None or num_pix < 2:
            raise ValueError("num_pix must be an integer >= 2")

        # Build grid limits: side length = num_pix * jwst_pix
        side = num_pix * jwst_pix
        x_min, x_max = center_x - side / 2.0, center_x + side / 2.0
        y_min, y_max = center_y - side / 2.0, center_y + side / 2.0

        # Axes (1D) and meshgrid (2D)
        x = np.linspace(x_min, x_max, num_pix)
        y = np.linspace(y_min, y_max, num_pix)
        xx, yy = np.meshgrid(x, y)  # shape (Ny, Nx)

        # Analytic lens model to generate the table
        lens_model = LensModel(['NFW_ELLIPSE_CSE'])
        kwargs_lens = [{
            'Rs': Rs_angle,
            'alpha_Rs': alpha_Rs,
            'e1': e1,
            'e2': e2,
            'center_x': lens_center_x,
            'center_y': lens_center_y
        }]

        # Compute absolute deflections on the grid
        ax_abs, ay_abs = lens_model.alpha(xx, yy, kwargs_lens)

        # Compute the center deflection once (absolute coordinates)
        ax_c, ay_c = lens_model.alpha(np.array([center_x]), np.array([center_y]), kwargs_lens)
        ax_c = float(ax_c[0])
        ay_c = float(ay_c[0])

        # Store *local* field: subtract the center deflection everywhere
        ax_loc = ax_abs - ax_c
        ay_loc = ay_abs - ay_c

        # Interpolators for the local field (RegularGridInterpolator uses (y, x) axis order)
        bounds_error = not extrapolate
        fill_value = None if extrapolate else 0.0
        self._interp_ax = RegularGridInterpolator(
            (y, x), ax_loc, bounds_error=bounds_error, fill_value=fill_value
        )
        self._interp_ay = RegularGridInterpolator(
            (y, x), ay_loc, bounds_error=bounds_error, fill_value=fill_value
        )

        # Save center for offsetting inputs in __call__
        self._center_x = center_x
        self._center_y = center_y

    def __call__(self, x, y):
        """
        Evaluate the *local* deflection field at offsets (x, y) relative to the
        stored center. Returns arrays shaped like np.asarray(x) and np.asarray(y).

        Specifically: alpha(center_x + x, center_y + y) - alpha(center_x, center_y).
        """
        x_arr = np.asarray(x) + self._center_x
        y_arr = np.asarray(y) + self._center_y

        pts = np.column_stack([y_arr.ravel(), x_arr.ravel()])  # (y, x) order
        ax = self._interp_ax(pts).reshape(np.shape(x_arr))
        ay = self._interp_ay(pts).reshape(np.shape(y_arr))
        return ax, ay
