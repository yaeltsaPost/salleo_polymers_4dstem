#Stephanie 4/9/2024
# Applies pixel size calibration

# Functions for creating flowline maps from diffraction spots
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import rgb_to_hsv
from matplotlib.colors import ListedColormap
from emdfile import tqdmnd, PointList, PointListArray
from scipy.signal import medfilt2d
import py4DSTEM
import numpy as np



def calibrate_bragg_peaks(bragg_peaks, rotation_calibration, step_size, \
                          inv_A_per_pixel_CL_corrected):
    bragg_peaks.calibration.set_R_pixel_size(step_size)
    bragg_peaks.calibration.set_R_pixel_units('nm')
    bragg_peaks.calibration.set_Q_pixel_size(inv_A_per_pixel_CL_corrected )
    bragg_peaks.calibration.set_Q_pixel_units('A^-1')
    bragg_peaks.calibration.set_QR_rotation(rotation_calibration) # added 
    bragg_peaks.calibrate()
    return bragg_peaks

#Stephanie 4/9/2024
# Compute the origin position pattern-by-pattern
def centering_bragg_peaks(bragg_peaks):
    origin_meas = bragg_peaks.measure_origin(
        # center_guess = (qx0,qy0),
        # score_method = "intensity weighted distance"
    )
    qx0_fit,qy0_fit,qx0_residuals,qy0_residuals = bragg_peaks.fit_origin(
    # fitfunction='bezier_two',
    # plot_range = 2.0,
    )
    bragg_peaks.calibration.set_origin((qx0_fit, qy0_fit))
    bragg_peaks.calibrate()
    return bragg_peaks
    #Stephanie 4/9/2024

# def filter_bragg_peaks(bragg_peaks,q_range,inv_A_per_pixel_CL_corrected,threshold,boolean_PI):
#     return bragg_peaks,boolean_PI

def remove_background(image_path):
    img = Image.open(image_path)
    img = img.convert("RGBA")

    datas = img.getdata()

    new_data = []
    for item in datas:
        # If the pixel is almost black (adjust values according to your image)
        if item[0] < 10 and item[1] < 10 and item[2] < 10:
            new_data.append((255, 255, 255, 0))  # Set fully transparent for black pixels
        else:
            new_data.append(item)

    img.putdata(new_data)
    return img

def orientation_correlation(
    orient_hist,
    radius_max=None,
    progress_bar=True,
):
    """
    Take in the 4D orientation histogram, and compute the distance-angle (auto)correlations

    Args:
        orient_hist (array):    3D or 4D histogram of all orientations with coordinates [x y radial_bin theta]
        radius_max (float):     Maximum radial distance for correlogram calculation. If set to None, the maximum
                                radius will be set to min(orient_hist.shape[0],orient_hist.shape[1])/2.

    Returns:
        orient_corr (array):          3D or 4D array containing correlation images as function of (dr,dtheta)
    """

    # Array sizes
    size_input = np.array(orient_hist.shape)
    if radius_max is None:
        radius_max = np.ceil(np.min(orient_hist.shape[1:3]) / 2).astype("int")
    size_corr = np.array(
        [
            np.maximum(2 * size_input[1], 2 * radius_max),
            np.maximum(2 * size_input[2], 2 * radius_max),
        ]
    )

    # Initialize orientation histogram
    orient_hist_pad = np.zeros(
        (
            size_input[0],
            size_corr[0],
            size_corr[1],
            size_input[3],
        ),
        dtype="complex",
    )
    orient_norm_pad = np.zeros(
        (
            size_input[0],
            size_corr[0],
            size_corr[1],
        ),
        dtype="complex",
    )

    # Pad the histogram in real space
    x_inds = np.arange(size_input[1])
    y_inds = np.arange(size_input[2])
    orient_hist_pad[:, x_inds[:, None], y_inds[None, :], :] = orient_hist
    orient_norm_pad[:, x_inds[:, None], y_inds[None, :]] = np.sum(
        orient_hist, axis=3
    ) / np.sqrt(size_input[3])
    orient_hist_pad = np.fft.fftn(orient_hist_pad, axes=(1, 2, 3))
    orient_norm_pad = np.fft.fftn(orient_norm_pad, axes=(1, 2))

    # Radial coordinates for integration
    x = (
        np.mod(np.arange(size_corr[0]) + size_corr[0] // 2, size_corr[0])
        - size_corr[0] // 2
    )
    y = (
        np.mod(np.arange(size_corr[1]) + size_corr[1] // 2, size_corr[1])
        - size_corr[1] // 2
    )
    ya, xa = np.meshgrid(y, x)
    ra = np.sqrt(xa**2 + ya**2)

    # coordinate subset
    sub0 = ra <= radius_max
    sub1 = ra <= radius_max - 1
    rF0 = np.floor(ra[sub0]).astype("int")
    rF1 = np.floor(ra[sub1]).astype("int")
    dr0 = ra[sub0] - rF0
    dr1 = ra[sub1] - rF1
    inds = np.concatenate((rF0, rF1 + 1))
    weights = np.concatenate((1 - dr0, dr1))

    # init output
    num_corr = (0.5 * size_input[0] * (size_input[0] + 1)).astype("int")
    orient_corr = np.zeros(
        (
            num_corr,
            (size_input[3] // 2 + 1).astype("int"),
            radius_max + 1,
        )
    )

    # Main correlation calculation
    ind_output = 0
    for a0, a1 in tqdmnd(
        range(size_input[0]),
        range(size_input[0]),
        desc="Calculate correlation plots",
        unit=" probe positions",
        disable=not progress_bar,
    ):
        # for a0 in range(size_input[0]):
        #     for a1 in range(size_input[0]):
        if a0 <= a1:
            # Correlation
            c = np.real(
                np.fft.ifftn(
                    orient_hist_pad[a0, :, :, :]
                    * np.conj(orient_hist_pad[a1, :, :, :]),
                    axes=(0, 1, 2),
                )
            )

            # Loop over all angles from 0 to pi/2  (half of indices)
            for a2 in range((size_input[3] / 2 + 1).astype("int")):
                orient_corr[ind_output, a2, :] = np.bincount(
                    inds,
                    weights=weights
                    * np.concatenate((c[:, :, a2][sub0], c[:, :, a2][sub1])),
                    minlength=radius_max,
                )

            # normalize
            c_norm = np.real(
                np.fft.ifftn(
                    orient_norm_pad[a0, :, :] * np.conj(orient_norm_pad[a1, :, :]),
                    axes=(0, 1),
                )
            )
            sig_norm = np.bincount(
                inds,
                weights=weights * np.concatenate((c_norm[sub0], c_norm[sub1])),
                minlength=radius_max,
            )
            orient_corr[ind_output, :, :] /= sig_norm[None, :]

            # increment output index
            ind_output += 1

    return orient_corr

def remove_hot_pixels_aux(dataset, relative_threshold=0.8 ):
    mask_hot_pixels =  dataset.tree('dp_mean').data - medfilt2d(dataset.tree('dp_mean').data) > relative_threshold * dataset.tree('dp_mean').data
    print('Total hot pixels = ' + str(mask_hot_pixels.sum()))
    
    # py4DSTEM.show(
    #     mask_hot_pixels,
    #     figsize=(6,6),
    # )
    # Apply mask - this step is not reversible!
    dataset.data *= (1 - mask_hot_pixels[None,None,:,:].astype('uint8'))
    dataset.get_dp_mean();
    dataset.get_dp_max();
    py4DSTEM.show(
        dataset.tree('dp_mean'),
        figsize=(6,6),
    )
    return dataset, mask_hot_pixels

def plot_orientation_correlation(
    orient_corr,
    prob_range=[0.1, 10.0],
    calculate_coefs=False,
    fraction_coefs=0.5,
    length_fit_slope=10,
    plot_overlaid_coefs=True,
    inds_plot=None,
    pixel_size=None,
    pixel_units=None,
    fontsize=24,
    figsize=(8, 6),
    returnfig=False,
    title=True
):
    """
    Plot the distance-angle (auto)correlations in orient_corr.

    Parameters
    ----------
    orient_corr (array):
        3D or 4D array containing correlation images as function of (dr,dtheta)
        1st index represents each pair of rings.
    prob_range (array):
        Plotting range in units of "multiples of random distribution".
    calculate_coefs (bool):
        If this value is True, the 0.5 and 0.1 distribution fraction of the
        radial and annular correlations will be calculated and printed.
    fraction_coefs (float):
        What fraction to calculate the correlation distribution coefficients for.
    length_fit_slope (int):
        Number of pixels to fit the slope of angular vs radial intercept.
    plot_overlaid_coefs (bool):
        If this value is True, the 0.5 and 0.1 distribution fraction of the
        radial and annular correlations will be overlaid onto the plots.
    inds_plot (float):
        Which indices to plot for orient_corr.  Set to "None" to plot all pairs.
    pixel_size (float):
        Pixel size for x axis.
    pixel_units (str):
        units of pixels.
    fontsize (float):
        Font size.  Title will be slightly larger, axis slightly smaller.
    figsize (array):
        Size of the figure panels.
    returnfig (bool):
        Set to True to return figure axes.

    Returns
    --------
    fig, ax (handles):
        Figure and axes handles (optional).

    """

    # Make sure range is an numpy array
    prob_range = np.array(prob_range)

    if pixel_size is None:
        pixel_size = 1
    if pixel_units is None:
        pixel_units = "pixels"

    # Get the pair indices
    size_input = orient_corr.shape
    num_corr = (np.sqrt(8 * size_input[0] + 1) // 2 - 1 // 2).astype("int")
    ya, xa = np.meshgrid(np.arange(num_corr), np.arange(num_corr))
    keep = ya >= xa
    # row 0 is the first diff ring, row 1 is the second diff ring
    pair_inds = np.vstack((xa[keep], ya[keep]))

    if inds_plot is None:
        inds_plot = np.arange(size_input[0])
    elif np.ndim(inds_plot) == 0:
        inds_plot = np.atleast_1d(inds_plot)
    else:
        inds_plot = np.array(inds_plot)

    # Custom divergent colormap:
    # dark blue
    # light blue
    # white
    # red
    # dark red
    N = 256
    cvals = np.zeros((N, 4))
    cvals[:, 3] = 1
    c = np.linspace(0.0, 1.0, int(N / 4))

    cvals[0 : int(N / 4), 1] = c * 0.4 + 0.3
    cvals[0 : int(N / 4), 2] = 1

    cvals[int(N / 4) : int(N / 2), 0] = c
    cvals[int(N / 4) : int(N / 2), 1] = c * 0.3 + 0.7
    cvals[int(N / 4) : int(N / 2), 2] = 1

    cvals[int(N / 2) : int(N * 3 / 4), 0] = 1
    cvals[int(N / 2) : int(N * 3 / 4), 1] = 1 - c
    cvals[int(N / 2) : int(N * 3 / 4), 2] = 1 - c

    cvals[int(N * 3 / 4) : N, 0] = 1 - 0.5 * c
    new_cmap = ListedColormap(cvals)

    if calculate_coefs:
        # Perform fitting
        def fit_dist(x, *coefs):
            int0 = coefs[0]
            int_bg = coefs[1]
            sigma = coefs[2]
            p = coefs[3]
            return (int0 - int_bg) * np.exp(np.abs(x) ** p / (-1 * sigma**p)) + int_bg

    # plotting
    num_plot = inds_plot.shape[0]
    fig, ax = plt.subplots(num_plot, 1, figsize=(figsize[0], num_plot * figsize[1]))

    # loop over indices
    for count, ind in enumerate(inds_plot):
        if num_plot > 1:
            p = ax[count].imshow(
                np.log10(orient_corr[ind, :, :]),
                vmin=np.log10(prob_range[0]),
                vmax=np.log10(prob_range[1]),
                aspect="auto",
                cmap=new_cmap,
            )
            ax_handle = ax[count]
        else:
            p = ax.imshow(
                np.log10(orient_corr[ind, :, :]),
                vmin=np.log10(prob_range[0]),
                vmax=np.log10(prob_range[1]),
                aspect="auto",
                cmap=new_cmap,
            )
            ax_handle = ax
        #################################################################
        # cbar = fig.colorbar(p, ax=ax_handle)
        # t = cbar.get_ticks()
        # t_lab = []
        # for a1 in range(t.shape[0]):
        #     t_lab.append(f"{10**t[a1]:.2g}")

        # cbar.set_ticks(t)
        # cbar.ax.set_yticklabels(t_lab, fontsize=fontsize * 1.)
        # cbar.ax.set_ylabel("Probability (m.r.d.)", fontsize=fontsize)

        ind_0 = pair_inds[0, ind]
        ind_1 = pair_inds[1, ind]

        if ind_0 != ind_1 and title==True:
            ax_handle.set_title(
                "Correlation of Rings " + str(ind_0) + " and " + str(ind_1),
                fontsize=fontsize * 1.,###########################################################
            )
        else:
            if title==True:
                ax_handle.set_title(
                    "Autocorrelation of Ring " + str(ind_0), fontsize=fontsize * 1.
            )

        # x axis labels
        # if pixel_size is not None:
        # x_t = ax_handle.get_xticks()
        # sub = np.logical_or(x_t < 0, x_t > orient_corr.shape[2])
        # x_t_new = np.delete(x_t, sub)
        # ax_handle.set_xticks(x_t_new)
        # ax_handle.set_xticklabels(x_t_new * pixel_size, fontsize=fontsize * 1.)
        # ax_handle.set_xlabel("Radial Distance (" + pixel_units + ")", fontsize=fontsize)

        # x axis labels
        # if pixel_size is not None:
        x_t = ax_handle.get_xticks()
        sub = np.logical_or(x_t < 0, x_t > orient_corr.shape[2])
        x_t_new = np.delete(x_t, sub)
        x_t_new_int = np.round(x_t_new * pixel_size).astype(int)
        ax_handle.set_xticks(x_t_new)
        ax_handle.set_xticklabels(x_t_new_int, fontsize=fontsize * 1.)
        ax_handle.set_xlabel("Radial Distance (" + pixel_units + ")", fontsize=fontsize)

        # y axis labels
        ax_handle.invert_yaxis()
        ax_handle.set_ylabel("Relative Orientation (°)", fontsize=fontsize)
        y_ticks = np.linspace(0, orient_corr.shape[1] - 1, 10, endpoint=True)
        ax_handle.set_yticks(y_ticks)
        ax_handle.set_yticklabels(
            ["0", "", "", "30", "", "", "60", "", "", "90"], fontsize=fontsize * 1.
        )

        if calculate_coefs:
            # Radial fractions
            y = np.arange(orient_corr.shape[2])
            if orient_corr[ind, 0, 0] > orient_corr[ind, -1, 0]:
                z = orient_corr[ind, 0, :]
            else:
                z = orient_corr[ind, -1, :]
            coefs = [np.max(z), np.min(z), y[-1] * 0.25, 2]
            bounds = ((1e-3, 0, 1e-3, 1.0), (np.inf, np.inf, np.inf, np.inf))
            coefs = curve_fit(fit_dist, y, z, p0=coefs, bounds=bounds)[0]
            coef_radial = coefs[2] * (np.log(1 / fraction_coefs) ** (1 / coefs[3]))

            # Annular fractions
            x = np.arange(orient_corr.shape[1])
            if orient_corr[ind, 0, 0] > orient_corr[ind, -1, 0]:
                z = orient_corr[ind, :, 0]
            else:
                z = np.flip(orient_corr[ind, :, 0], axis=0)
            z = np.maximum(z, 1.0)
            coefs = [np.max(z), np.min(z), x[-1] * 0.25, 2]
            bounds = ((1e-3, 0, 1e-3, 1.0), (np.inf, np.inf, np.inf, np.inf))
            coefs = curve_fit(fit_dist, x, z, p0=coefs, bounds=bounds)[0]
            coef_annular = coefs[2] * (np.log(1 / fraction_coefs) ** (1 / coefs[3]))
            if orient_corr[ind, 0, 0] <= orient_corr[ind, -1, 0]:
                coef_annular = orient_corr.shape[1] - 1 - coef_annular
            pixel_size_annular = 90 / (orient_corr.shape[1] - 1)

            # Slope of annular vs radial correlations as radius --> 0
            x_slope = np.argmin(
                np.abs(orient_corr[ind, :, :length_fit_slope] - 1.0), axis=0
            )
            y_slope = np.arange(length_fit_slope)
            coefs_slope = np.polyfit(y_slope, x_slope, 1)

            # Print results
            if ind_0 != ind_1:
                print("Correlation of Rings " + str(ind_0) + " and " + str(ind_1))
            else:
                print("Autocorrelation of Ring " + str(ind_0))
            print(
                str(np.round(fraction_coefs * 100).astype("int"))
                + "% probability radial distance = "
                + str(np.round(coef_radial * pixel_size, 0))
##########                + " "
                + pixel_units
            )
            print(
                str(np.round(fraction_coefs * 100).astype("int"))
                + "% probability annular distance = "
                + str(np.round(coef_annular * pixel_size_annular, 0))
                + " degrees"
            )
            print(
                "slope = "
                + str(np.round(coefs_slope[0] * pixel_size_annular / pixel_size, 0))
                + " degrees/"
                + pixel_units
            )
            print()

        if plot_overlaid_coefs:
            if num_plot > 1:
                ax_handle = ax[count]
            else:
                ax_handle = ax

            if orient_corr[ind, 0, 0] > orient_corr[ind, -1, 0]:
                ax_handle.plot(
                    np.array((coef_radial, coef_radial, 0.0)),
                    np.array((0.0, coef_annular, coef_annular)),
                    color=(1.0, 1.0, 1.0),
                )
                ax_handle.plot(
                    np.array((coef_radial, coef_radial, 0.0)),
                    np.array((0.0, coef_annular, coef_annular)),
                    color=(0.0, 0.0, 0.0),
                    linestyle="--",
                )
            else:
                ax_handle.plot(
                    np.array((coef_radial, coef_radial, 0.0)),
                    np.array(
                        (
                            orient_corr.shape[1] - 1,
                            coef_annular,
                            coef_annular,
                        )
                    ),
                    color=(1.0, 1.0, 1.0),
                )
                ax_handle.plot(
                    np.array((coef_radial, coef_radial, 0.0)),
                    np.array(
                        (
                            orient_corr.shape[1] - 1,
                            coef_annular,
                            coef_annular,
                        )
                    ),
                    color=(0.0, 0.0, 0.0),
                    linestyle="--",
                )
            ax_handle.plot(
                y_slope,
                y_slope.astype("float") * coefs_slope[0] + coefs_slope[1],
                color=(0.0, 0.0, 0.0),
                linestyle="--",
            )

    # Fix spacing
    fig.tight_layout(pad=1.0)

    if returnfig:
        return fig, ax
    plt.show()


def get_intensity(orient, x, y, t):
    # utility function to get histogram intensites

    x = np.clip(x, 0, orient.shape[0] - 2)
    y = np.clip(y, 0, orient.shape[1] - 2)

    xF = np.floor(x).astype("int")
    yF = np.floor(y).astype("int")
    tF = np.floor(t).astype("int")
    dx = x - xF
    dy = y - yF
    dt = t - tF
    t1 = np.mod(tF, orient.shape[2])
    t2 = np.mod(tF + 1, orient.shape[2])

    int_vals = (
        orient[xF, yF, t1] * ((1 - dx) * (1 - dy) * (1 - dt))
        + orient[xF, yF, t2] * ((1 - dx) * (1 - dy) * (dt))
        + orient[xF, yF + 1, t1] * ((1 - dx) * (dy) * (1 - dt))
        + orient[xF, yF + 1, t2] * ((1 - dx) * (dy) * (dt))
        + orient[xF + 1, yF, t1] * ((dx) * (1 - dy) * (1 - dt))
        + orient[xF + 1, yF, t2] * ((dx) * (1 - dy) * (dt))
        + orient[xF + 1, yF + 1, t1] * ((dx) * (dy) * (1 - dt))
        + orient[xF + 1, yF + 1, t2] * ((dx) * (dy) * (dt))
    )

    return int_vals
def set_intensity(orient, xy_t_int):
    # utility function to set flowline intensites

    xF = np.floor(xy_t_int[:, 0]).astype("int")
    yF = np.floor(xy_t_int[:, 1]).astype("int")
    tF = np.floor(xy_t_int[:, 2]).astype("int")
    dx = xy_t_int[:, 0] - xF
    dy = xy_t_int[:, 1] - yF
    dt = xy_t_int[:, 2] - tF

    inds_1D = np.ravel_multi_index(
        [xF, yF, tF], orient.shape[0:3], mode=["clip", "clip", "wrap"]
    )
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (1 - dx) * (
        1 - dy
    ) * (1 - dt)
    inds_1D = np.ravel_multi_index(
        [xF, yF, tF + 1], orient.shape[0:3], mode=["clip", "clip", "wrap"]
    )
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (1 - dx) * (
        1 - dy
    ) * (dt)
    inds_1D = np.ravel_multi_index(
        [xF, yF + 1, tF], orient.shape[0:3], mode=["clip", "clip", "wrap"]
    )
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (1 - dx) * (
        dy
    ) * (1 - dt)
    inds_1D = np.ravel_multi_index(
        [xF, yF + 1, tF + 1], orient.shape[0:3], mode=["clip", "clip", "wrap"]
    )
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (1 - dx) * (
        dy
    ) * (dt)
    inds_1D = np.ravel_multi_index(
        [xF + 1, yF, tF], orient.shape[0:3], mode=["clip", "clip", "wrap"]
    )
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (dx) * (
        1 - dy
    ) * (1 - dt)
    inds_1D = np.ravel_multi_index(font
        [xF + 1, yF, tF + 1], orient.shape[0:3], mode=["clip", "clip", "wrap"]
    )
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (dx) * (
        1 - dy
    ) * (dt)
    inds_1D = np.ravel_multi_index(
        [xF + 1, yF + 1, tF], orient.shape[0:3], mode=["clip", "clip", "wrap"]
    )
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (dx) * (dy) * (
        1 - dt
    )
    inds_1D = np.ravel_multi_index(
        [xF + 1, yF + 1, tF + 1], orient.shape[0:3], mode=["clip", "clip", "wrap"]
    )
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (dx) * (dy) * (
        dt
    )

    return orient


def remove_background(image_path):
    img = Image.open(image_path)
    img = img.convert("RGBA")

    datas = img.getdata()

    new_data = []
    for item in datas:
        # If the pixel is almost black (adjust values according to your image)
        if item[0] < 10 and item[1] < 10 and item[2] < 10:
            new_data.append((255, 255, 255, 0))  # Set fully transparent for black pixels
        else:
            new_data.append(item)

    img.putdata(new_data)
    return img
#############
# def separate_peak_detection():
#     return 

# def separte_syntetic_robe(probe_semiangle,probe_width_backbone ):
#     probe_radius_backbone = round(probe_semiangle)
#     probe_width_backbone = 1.
    
#     # Make a synthetic probe
#     probe_init_backbone = py4DSTEM.Probe.generate_synthetic_probe(
#         radius = probe_radius_backbone,
#         width = probe_width_backbone,
#         Qshape = dataset.Qshape,
#     )
    
    probe_kernel_backbone = py4DSTEM.Probe.get_probe_kernel_edge_sigmoid(
        probe_init_backbone,
        (probe_radius_backbone*1.,probe_radius_backbone*3.5),#4
    )
    
    py4DSTEM.visualize.show_kernel(
        probe_kernel_backbone,
        R = 20,
        L = 20,
        W = 1
)
    return 
