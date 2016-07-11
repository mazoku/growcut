__author__ = 'tomas'

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# import tools
import numpy as np
import computational_core as cc

import scipy.ndimage.filters as scindifil

import skimage.transform as skitra

from PyQt4 import QtGui
from viewer_3D import Viewer_3D
from growcut import GrowCut

import sys
import os
if os.path.exists('../../imtools/'):
    # sys.path.append('../imtools/')
    sys.path.insert(0, '../../imtools/')
    from imtools import tools, misc
else:
    print 'You need to import package imtools: https://github.com/mjirik/imtools'
    sys.exit(0)


def run(data, mask=None, save_fig=False, smoothing=False, return_all=False, show=False, show_now=True, verbose=True):
    orig_shape = data.shape
    data = skitra.rescale(data, 0.5, preserve_range=True).astype(np.uint8)

    if data.ndim == 2:
        data = np.expand_dims(data, 0)
        if mask is not None:
            mask = np.expand_dims(mask, 0)
    # data = tools.smoothing_tv(data, weight=0.05, sliceId=0)
    # data = tools.smoothing_gauss(data, sigma=1, sliceId=0)
    if smoothing:
        data = tools.smoothing(data, sigmaSpace=10, sigmaColor=10, sliceId=0)

    # cc.hist2gmm(data, debug=verbose)
    # cc.gmm_segmentation(data, debug=verbose)
    # liver_rv = cc.estim_liver_prob_mod(data, show=False, show_now=show_now)
    _, liver_rv = tools.dominant_class(data, dens_min=10, dens_max=245, show=False, show_now=False)

    liver_prob = liver_rv.pdf(data)
    # app = QtGui.QApplication(sys.argv)
    # viewer = Viewer_3D(data)
    # viewer.show()
    # viewer2 = Viewer_3D(liver_prob, range=True)
    # viewer2.show()
    # sys.exit(app.exec_())

    # plt.figure()
    # plt.imshow(liver_prob[0,...], 'gray', interpolation='nearest')
    # plt.show()

    prob_c = 0.2
    prob_c_2 = 0.01
    seeds1 = liver_prob > (liver_prob.max() * prob_c)
    seeds2 = liver_prob <= (np.median(liver_prob) * prob_c_2)
    seeds = seeds1 + 2 * seeds2

    # plt.figure()
    # liver_prob = liver_prob[0,...]
    # plt.subplot(231), plt.imshow(liver_prob <= liver_prob.mean(), 'gray', interpolation='nearest'), plt.title('mean')
    # plt.subplot(232), plt.imshow(liver_prob <= liver_prob.min(), 'gray', interpolation='nearest'), plt.title('min')
    # plt.subplot(233), plt.imshow(liver_prob <= np.median(liver_prob), 'gray', interpolation='nearest'), plt.title('median')
    # plt.subplot(234), plt.imshow(liver_prob <= 0.5 * np.median(liver_prob), 'gray', interpolation='nearest'), plt.title('0.5 median')
    # plt.subplot(235), plt.imshow(liver_prob <= 0, 'gray', interpolation='nearest'), plt.title('0.1 median')
    # plt.show()

    # plt.figure()
    # plt.subplot(121), plt.imshow(liver_prob[0,...], 'gray', interpolation='nearest')
    # plt.subplot(122), plt.imshow(seeds[0,...], 'gray', interpolation='nearest')
    # plt.show()

    gc = GrowCut(data, seeds, smooth_cell=False, enemies_T=0.7)
    gc.run()

    labs = gc.get_labeled_im().astype(np.uint8)
    # labs2 = skifil.median(labs[0,...], selem=np.ones((3, 3)))
    labs_f = scindifil.median_filter(labs, size=3)

    # plt.figure()
    # plt.subplot(121), plt.imshow(labs[0,...], 'jet', interpolation='nearest')
    # plt.subplot(122), plt.imshow(labs_f[0,...], 'jet', interpolation='nearest')
    # plt.show()

    labs = labs_f

    plt.figure()
    plt.subplot(131), plt.imshow(data[0,...], 'gray', interpolation='nearest')
    plt.subplot(132), plt.imshow(seeds[0,...], 'jet', interpolation='nearest')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(cax=cax, ticks=np.unique(seeds))
    plt.subplot(133), plt.imshow(labs[0,...], 'jet', interpolation='nearest', vmin=0)
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(cax=cax, ticks=np.unique(seeds))
    plt.show()

    # if show:
    #     plt.show()


################################################################################
################################################################################
if __name__ == '__main__':

    # 2 hypo, 1 on the border --------------------
    # arterial 0.6mm - bad
    # fname = '/home/tomas/Data/medical/liver_segmentation/org-exp_183_46324212_arterial_0.6_B30f-.pklz'
    # venous 0.6mm - good
    # fname = '/home/tomas/Data/medical/liver_segmentation/org-exp_183_46324212_venous_0.6_B20f-.pklz'
    # venous 5mm - ok, but wrong approach
    fname = '/home/tomas/Data/medical/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'

    # hypo in venous -----------------------
    # arterial - bad
    # fname = '/home/tomas/Data/medical/liver_segmentation/org-exp_186_49290986_venous_0.6_B20f-.pklz'
    # venous - good
    # fname = '/home/tomas/Data/medical/liver_segmentation/org-exp_186_49290986_arterial_0.6_B20f-.pklz'

    # hyper, 1 on the border -------------------
    # arterial 0.6mm - not that bad
    # fname = '/home/tomas/Data/medical/liver_segmentation/org-exp_239_61293268_DE_Art_Abd_0.75_I26f_M_0.5-.pklz'
    # venous 5mm - bad
    # fname = '/home/tomas/Data/medical/liver_segmentation/org-exp_239_61293268_DE_Ven_Abd_0.75_I26f_M_0.5-.pklz'

    # shluk -----------------
    # arterial 5mm
    # fname = '/home/tomas/Data/medical/liver_segmentation/org-exp_180_49509315_arterial_5.0_B30f-.pklz'
    # fname = '/home/tomas/Data/medical/liver_segmentation/org-exp_180_49509315_venous_5.0_B30f-.pklz'

    # targeted

    # arterial 0.6mm - bad
    # fname = '/home/tomas/Data/medical/liver_segmentation/org-exp_238_54280551_Abd_Arterial_0.75_I26f_3-.pklz'
    # venous 0.6mm - bad
    # fname = '/home/tomas/Data/medical/liver_segmentation/org-exp_238_54280551_Abd_Venous_0.75_I26f_3-.pklz'

    debug = True
    verbose = True
    data_fname = '/home/tomas/Data/medical/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    data, mask, voxel_size = tools.load_pickle_data(data_fname)

    slice_ind = 17
    data_s = data[slice_ind, :, :]
    data_s = tools.windowing(data_s)
    mask_s = mask[slice_ind, :, :]
    # data_s = tools.windowing(data)
    # mask_s = mask

    run(data_s, mask=mask_s, smoothing=True, save_fig=False, show=True, verbose=verbose)