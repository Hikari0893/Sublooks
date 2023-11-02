import struct
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from scipy import signal, special


def threshold(X, m=0, M=None):
    """
    Warning : it works by side effect
    :param X: input data
    :param m: minimum
    :param M: maximum
    :return: X
    """

    if M is None:
        M = 3 * np.mean(X)
    X[X < m] = m
    X[X > M] = M
    return X


def T_EF(X, l, h, m, M):
    """ Element-wise homothety-translation for X between E=[h,l] to F=[m,M] """
    assert l < h and m <= M
    return (X - l) * (M - m) / (h - l) + m





####################################
### Data save and load functions ###
####################################

def cos2mat(imgName):
    """ Fuction copied from https://gitlab.telecom-paris.fr/ring/MERLIN/load_cosar.py """
    # print('Converting CoSAR to numpy array of size [ncolumns,nlines,2]')
    try:
        fin = open(imgName, 'rb')
    except IOError:
        legx = imgName + ': it is a not openable file'
        print(legx)
        print(u'failed to call cos2mat')
        return 0, 0, 0, 0
    ibib = struct.unpack(">i", fin.read(4))[0]
    irsri = struct.unpack(">i", fin.read(4))[0]
    irs = struct.unpack(">i", fin.read(4))[0]
    ias = struct.unpack(">i", fin.read(4))[0]
    ibi = struct.unpack(">i", fin.read(4))[0]
    irtnb = struct.unpack(">i", fin.read(4))[0]
    itnl = struct.unpack(">i", fin.read(4))[0]
    nlig = struct.unpack(">i", fin.read(4))[0]
    ncoltot = int(irtnb / 4)
    ncol = ncoltot - 2
    nlig = ias
    # print(u'Reading image in CoSAR format.  ncolumns=%d  nlines=%d' % (ncol, nlig))
    firm = np.zeros(4 * ncoltot, dtype=np.byte())
    imgcxs = np.empty([nlig, ncol], dtype=np.complex64())
    fin.seek(0)
    firm = fin.read(4 * ncoltot)
    firm = fin.read(4 * ncoltot)
    firm = fin.read(4 * ncoltot)
    firm = fin.read(4 * ncoltot)
    for iut in range(nlig):
        firm = fin.read(4 * ncoltot)
        imgligne = np.ndarray(2 * ncoltot, '>h', firm)
        imgcxs[iut, :] = imgligne[4:2 * ncoltot:2] + 1j * imgligne[5:2 * ncoltot:2]

    # print('[:,:,0] contains the real part of the SLC image data')
    # print('[:,:,1] contains the imaginary part of the SLC image data')

    # plt.hist((np.abs(imgcxs)[::5,::5]).flatten(),bins=500)
    # plt.imshow(np.abs(imgcxs)[::5,::5], cmap='gray')
    # plt.colorbar()
    # plt.show()

    return np.stack((np.real(imgcxs), np.imag(imgcxs)), axis=2)






def plot_Re_or_Im_amp(I_Re_or_Im, I_pred, PREPROCESSED, suptitle=None, Org_m=None, Org_M=None, PLOT=True):
    """
    :param I_Re_or_Im: squared real or imagniary part of an SLC
    """
    I_Re_or_Im = np.abs(I_Re_or_Im)
    I_pred = np.abs(I_pred)
    if PLOT:
        vmax = np.mean(I_Re_or_Im) + 3 * np.mean(I_Re_or_Im)
        plt.figure(figsize=(12, 10))
        plt.subplot(221);
        plt.imshow(I_Re_or_Im, vmax=vmax, cmap='gray');
        plt.title("I_Re_or_Im : True amplitude of real or imaginary image")
        plt.subplot(222);
        plt.imshow(I_pred, vmax=vmax, cmap='gray');
        plt.title("I_Re_or_Im_pred : Recovered amplitude of real or imaginary part")
        plt.subplot(223);
        plt.hist(I_Re_or_Im.flatten(), bins=256);
        plt.title("A's histogram\nmin=%.2f max=%.2f mean=%.2f std=%.2f" % (
        np.min(I_Re_or_Im), np.max(I_Re_or_Im), np.mean(I_Re_or_Im), np.std(I_Re_or_Im)));
        plt.xlabel("Values");
        plt.ylabel("Count")
        plt.subplot(224);
        plt.hist(I_pred.flatten(), bins=256);
        plt.title("A_pred's histogram\nmin=%.2f max=%.2f mean=%.2f std=%.2f" % (
        np.min(I_pred), np.max(I_pred), np.mean(I_pred), np.std(I_pred)));
        plt.xlabel("Values");
        plt.ylabel("Count")
        if not (suptitle is None):
            plt.suptitle(suptitle)
        plt.tight_layout()
    pass



def plot_residues(I, I_pred, res_only=False, suptitle=None):
    """ WARNING : some values are excluded with a theshold """
    Res_I = np.divide(I + 1e-10, I_pred + 1e-10)
    # Res_I = threshold(Res_I,0,np.percentile(Res_I,100-0.05)) # to exclude the 33 highest valus in the event of a big division
    # Res_I = threshold(Res_I, 0, 4)
    Res_I = np.clip(Res_I, 0, 4)
    if res_only:
        plt.figure(figsize=(12, 15))
        # plt.subplot(111, );
        plt.imshow(Res_I, cmap="gray");
        plt.title(r"Residual obtained from $I/I_{pred}$");
        plt.colorbar()
        if not (suptitle is None):
            plt.suptitle(suptitle)
        plt.tight_layout()
    else:
        plt.figure(figsize=(12, 15))
        plt.subplot(211, );
        plt.imshow(Res_I, cmap="gray");
        plt.title(r"Residual obtained from $I/I_{pred}$");
        plt.colorbar()
        plt.subplot(212);
        plt.hist(Res_I.flatten(), bins=256);
        plt.title("Residual's histogram\nmin=%.2f max=%.2f mean=%.2f std=%.2f" % (
        np.min(Res_I), np.max(Res_I), np.mean(Res_I), np.std(Res_I)));
        plt.xlabel("Values");
        plt.ylabel("Count")
        if not (suptitle is None):
            plt.suptitle(suptitle)
        plt.tight_layout()
    return Res_I


def threshold_and_clip(noisy, im, threshold=None):
    if threshold is None:
        threshold = np.mean(noisy) + 3*np.std(noisy)
    im = np.clip(im, 0, threshold)
    im = im / threshold * 255
    return im


#######################
### Tests functions ###
#######################


def image_bytescale_hist(image, nb_bins=256 // 10):
    # Plot
    I_plot = bytescale(image, np.min(image), 3 * np.mean(image), high=255, low=0)
    fig, axes = plt.subplots(1, 2)
    plt.subplot(121)
    myplot1 = plt.imshow(I_plot, cmap="gray")
    plt.title("Current image")
    plt.subplot(122)
    plt.hist(I_plot.flatten(), density=True, bins=nb_bins)
    plt.title("Histogram (density)")
    # Creating slider and update function
    axvalmax = plt.axes([0.2, 0.05, 0.25, 0.03])
    slider_valmax = Slider(axvalmax, "Val_max", 0.1, 7, valinit=3, valstep=0.1)

    def update(xr32):
        val = slider_valmax.val
        data = bytescale(image, np.min(image), val * np.mean(image), high=255, low=0)
        # print("The smallest value corresonding to 255 is %.1f"%(val*np.mean(image)))
        myplot1.set_data(data)
        axes[1].clear()
        axes[1].hist(data.flatten(), density=True, bins=nb_bins)
        plt.subplot(122)
        plt.title("Histogram (density)")
        fig.canvas.draw_idle()

    slider_valmax.on_changed(update)
    plt.show()
    pass


def image_logscale_hist(image, nb_bins=256 // 10):
    # Plot
    image_scaled = bytescale(image, np.min(image), 3 * np.mean(image), high=255, low=0)
    I_plot = np.log(1 + image_scaled.astype(float))
    I_plot = bytescale(I_plot, np.min(I_plot), np.max(I_plot), high=255, low=0)
    fig, axes = plt.subplots(1, 2)
    plt.subplot(121)
    myplot1 = plt.imshow(I_plot, cmap="gray")
    plt.title("Current image")
    plt.subplot(122)
    plt.hist(I_plot.flatten(), density=True, bins=nb_bins)
    plt.title("Histogram (density)")
    # Creating slider and update function
    axcstlog = plt.axes([0.2, 0.05, 0.25, 0.03])
    slider_cstlog = Slider(axcstlog, r"x as in $log(10^x+I)$", -15, 15, valinit=1, valstep=1)

    def update(xr32):
        val = slider_cstlog.val
        img = np.log(10 ** val + image_scaled)
        data = bytescale(img, np.min(img), np.max(img), high=255, low=0)
        # print("The smallest value corresonding to 255 is %.1f"%(val*np.mean(image)))
        myplot1.set_data(data)
        axes[1].clear()
        axes[1].hist(data.flatten(), density=True, bins=nb_bins)
        plt.subplot(122)
        plt.title("Histogram (density)")
        fig.canvas.draw_idle()

    slider_cstlog.on_changed(update)
    plt.show()
    pass

def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    :author: Andrea Pulella <andrea.pulella@dlr.de>
    data : ndarray. Input array.
    cmin : scalar, optional. Bias scaling of small values. Default is data.min().
    cmax : scalar, optional. Bias scaling of large values. Default is data.max().
    high : scalar, optional. Scale max value to `high`.
    low : scalar, optional. Scale min value to `low`.
    :returns: img_array : uint8 ndarray. The byte-scaled array.
    """
    ### Nils: commented out this section
    # if data.dtype == np.uint8:
    #     return data

    if high < low:
        raise ValueError("`high` should be larger than `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data * 1.0 - cmin) * scale + 0.4999
    bytedata[bytedata > high] = high
    bytedata[bytedata < low] = low  # Nils changed : ... <0] = 0
    return np.cast[np.uint8](bytedata) + np.cast[np.uint8](low)
