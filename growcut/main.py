__author__ = 'tomas'

import matplotlib.pyplot as plt

# import tools
import numpy as np
import computational_core as cc

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

    if data.ndim == 2:
        data = np.expand_dims(data, 0)
        if mask is not None:
            mask = np.expand_dims(mask, 0)
    # data = tools.smoothing_tv(data, weight=0.05, sliceId=0)
    # data = tools.smoothing_gauss(data, sigma=1, sliceId=0)
    if smoothing:
        data = tools.smoothing(data, sigmaSpace=5, sigmaColor=0.02, sliceId=0)

    # cc.hist2gmm(data, debug=verbose)
    # cc.gmm_segmentation(data, debug=verbose)
    liver_rv = cc.estim_liver_prob_mod(data, show=False, show_now=show_now)

    liver_prob = liver_rv.pdf(data)
    # app = QtGui.QApplication(sys.argv)
    # viewer = Viewer_3D(data)
    # viewer.show()
    # viewer2 = Viewer_3D(liver_prob, range=True)
    # viewer2.show()
    # sys.exit(app.exec_())

    prob_c = 0.2
    seeds = liver_prob > (liver_prob.max() * prob_c)

    plt.figure()
    plt.subplot(121), plt.imshow(liver_prob[0,...], 'gray', interpolation='nearest')
    plt.subplot(122), plt.imshow(seeds[0,...], 'gray', interpolation='nearest')
    plt.show()

    gc = GrowCut(data, seeds + 1)

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