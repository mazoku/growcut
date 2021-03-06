from __future__ import division

__author__ = 'Ryba'

import numpy as np
import Tkinter as tk
import ttk
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
import cv2
import progressbar
import io3d

import skimage.data as skidat
import skimage.io as skiio
import skimage.exposure as skiexp
import skimage.transform as skitra
import skimage.color as skicol
import skimage.morphology as skimor
import skimage.segmentation as skiseg

import sys
import os
if os.path.exists('/home/tomas/projects/imtools/'):
    sys.path.insert(0, '/home/tomas/projects/imtools/')
    from imtools import tools, misc
else:
    print 'Import error in growcut.py. You need to import package imtools: https://github.com/mjirik/imtools'
    # sys.exit(0)

if os.path.exists('../../thesis/'):
    # sys.path.append('../imtools/')
    sys.path.insert(0, '../../thesis/')
    import identify_liver_blob as ilb


class GrowCut:
    def __init__(self, data, seeds, maxits=50, smooth_cell=True, enemies_T=1., nghoodtype='sparse', gui=None, verbose=True):
        '''
        data ... input data; should be 3D in form [slices, rows, columns]
        seeds ... seed points; same shape as data; background should have label 1
        '''
        self.data = data.astype(np.float64)
        self.data = np.array(self.data, ndmin=3)
        # self.data = self.smoothing( self.data )
        self.seeds = seeds
        self.n_classes = self.seeds.max()
        self.maxits = maxits
        self.gui = gui
        self.verbose = verbose
        self.visfig = None
        self.smooth_cell = smooth_cell
        self.enemies_T = enemies_T

        self.nslices, self.nrows, self.ncols = self.data.shape
        self.npixels = self.nrows * self.ncols
        self.nvoxels = self.npixels * self.nslices

        if self.nslices == 1: #2D image
            if nghoodtype == 'sparse':
                self.nghood = 4
            elif nghoodtype == 'full':
                self.nghood = 8
        else: #3D image
            if nghoodtype == 'sparse':
                self.nghood = 6
            elif nghoodtype == 'full':
                self.nghood = 26

        self.seeds = np.reshape(self.seeds, (1, self.nvoxels)).squeeze()
        lind = np.ravel_multi_index(np.indices(self.data.shape), self.data.shape) #linear indices in array form
        self.lindv = np.reshape(lind, (1,self.nvoxels)).squeeze() #linear indices in vector form
        self.coordsv = np.array(np.unravel_index( self.lindv, self.data.shape)) #coords in array [dim * nvoxels]

        self.strengths = np.zeros_like(self.seeds, dtype=np.float64)
        self.strengths = np.where(self.seeds, 1., 0.)
        self.maxC = self.data.max() - self.data.min()
        # self.maxC = 441.673

        self.labels = self.seeds.copy()

        x = np.arange( 0, self.data.shape[2] )
        y = np.arange( 0, self.data.shape[1] )
        self.xgrid, self.ygrid = np.meshgrid( x, y )

        self.activePixs = np.argwhere(self.seeds > 0).squeeze()

    def run(self):
        self.neighborsM = self.make_neighborhood_matrix()

        converged = False
        it = 0
        if self.gui:
            self.gui.statusbar.config(text='Segmentation in progress...')
            self.gui.progressbar.set(0, '')

        # initialize progressbar
        if self.verbose:
            widgets = ["Iterations: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
            pbar = progressbar.ProgressBar(maxval=self.maxits, widgets=widgets)
            pbar.start()

        while (not converged) and (it < self.maxits):
            it += 1
            if self.verbose:
                pbar.update(it)
            # print 'iteration #%i' % it
            converged = self.iteration()

            if self.gui:
                self.redraw_fig()
                self.gui.progressbar.step( 1./self.maxits )
                self.gui.canvas.draw()

            if it == self.maxits and self.gui:
                self.gui.statusbar.config( text='Maximal number of iterations reached' )
                break

            if converged and self.gui:
                self.gui.statusbar.config( text='Algorithm converged after {0} iterations'.format(it) )
        if self.verbose:
            pbar.finish()

        # print 'done'
        # qq = np.reshape( self.labels, self.data.shape )
        # plt.figure(), plt.imshow( qq[0,:,:] ), plt.show()

    def smoothing( self, data ):
        for i in range( data.shape[0] ):
            data[i,:,:] = cv2.GaussianBlur( data[i,:,:], (3,3), 0 )
        return data

    def redraw_fig(self):
        plt.hold( False )
        plt.imshow( self.gui.img[self.gui.currFrameIdx,:,:], aspect='equal' )
        plt.hold( True )
        # for i in self.linesL:
        #     self.ax.add_artist( i )

        if (self.labels == 1).any():
            self.blend_image( np.reshape( self.labels, self.data.shape )[self.gui.currFrameIdx,:,:] == 1, color='b' )
        if (self.labels == 2).any():
            self.blend_image( np.reshape( self.labels, self.data.shape )[self.gui.currFrameIdx,:,:] == 2, color='r' )
        if (self.labels == 3).any():
            self.blend_image( np.reshape( self.labels, self.data.shape )[self.gui.currFrameIdx,:,:] == 3, color='g' )

        self.gui.fig.canvas.draw()

    def get_labeled_im(self):
        return np.reshape(self.labels, self.data.shape)

    def blend_image( self, im, color, alpha=0.5 ):
        plt.hold( True )
        layer = np.zeros( np.hstack( (im.shape, 4)), dtype=np.uint8 )
        imP = np.argwhere( im )

        if color == 'r':
            layer[ imP[:,0], imP[:,1], 0 ] = 255
        elif color == 'g':
            layer[ imP[:,0], imP[:,1], 1 ] = 255
        elif color == 'b':
            layer[ imP[:,0], imP[:,1], 2 ] = 255
        elif color == 'c':
            layer[ imP[:,0], imP[:,1], (1,2) ] = 255
        elif color == 'm':
            layer[ imP[:,0], imP[:,1], (0,2) ] = 255
        elif color == 'y':
            layer[ imP[:,0], imP[:,1], (0,1) ] = 255

        layer[:,:,3] = 255 * im #alpha channel
        plt.imshow( layer, alpha=alpha  )

    def draw_contours(self):
        qq = np.reshape( self.labels, self.data.shape )[self.gui.currFrameIdx,:,:]
        plt.contour( self.xgrid, self.ygrid, qq==1, levels = [1], colors = 'b')
        plt.contour( self.xgrid, self.ygrid, qq==2, levels = [1], colors = 'r')

        plt.draw()

    def iteration( self):
        converged = True
        labelsN = self.labels.copy()

        # warnings.warn( 'For speed up - maintain list of active pixels and iterate only over them. '
        #                'Active pixel is pixel, that changed label or strength.' )

        newbies = list()
        for p in self.activePixs:
            pcoords = tuple(self.coordsv[:, p])
            for q in range(self.nghood):
                nghbInd = self.neighborsM[q, p]
                if np.isnan(nghbInd):
                    continue
                else:
                    nghbInd = int(nghbInd)
                nghbcoords = tuple(self.coordsv[:, nghbInd])
                # with warnings.catch_warnings(record=True) as w:
                C = np.absolute(self.data[pcoords] - self.data[nghbcoords])
                # C = np.sqrt((self.data[pcoords] - self.data[nghbcoords]) ** 2)

                g = 1 - (C / self.maxC)

                #attack the neighbor
                #if g * self.strengths[nghbInd] > self.strengths[p]:
                if g * self.strengths[p] > self.strengths[nghbInd]:
                    self.strengths[nghbInd] = g * self.strengths[p]
                    labelsN[nghbInd] = self.labels[p]
                    newbies.append(nghbInd)
                    converged = False

        self.labels = labelsN
        self.activePixs = newbies

        # plt.figure()
        # plt.subplot(121), plt.imshow(self.get_labeled_im()[0,...], 'jet', interpolation='nearest', vmin=0, vmax=self.n_classes)
        # plt.subplot(122), plt.imshow(np.reshape(labelsN, self.data.shape)[0,...], 'jet', interpolation='nearest', vmin=0, vmax=self.n_classes)
        # plt.show()

        if self.smooth_cell:
            labelsN_s = self.cell_smoothing(labelsN.copy())

            # plt.figure()
            # plt.subplot(121), plt.imshow(np.reshape(labelsN, self.data.shape)[0,...], 'jet', interpolation='nearest', vmin=0, vmax=self.n_classes)
            # plt.subplot(122), plt.imshow(np.reshape(labelsN_s, self.data.shape)[0,...], 'jet', interpolation='nearest', vmin=0, vmax=self.n_classes)
            # plt.suptitle('# of changes: %i' % (labelsN != labelsN_s).sum())
            # plt.show()

            self.labels = labelsN_s

        return converged

    def cell_smoothing(self, data=None):
        # TODO: otestovat
        # self.labels = np.array([[1, 1, 1],[1, 2, 1],[1, 1, 1]]).flatten()
        if data is None:
            data = self.labels
        pts = np.argwhere(data > 0).squeeze()

        # pts_im = np.zeros_like(data)
        # pts_im[pts] = 1
        # plt.figure()
        # plt.subplot(121), plt.imshow(np.reshape(data, self.data.shape)[0,...], 'jet')
        # plt.subplot(122), plt.imshow(np.reshape(pts_im, self.data.shape)[0,...], 'gray')
        # plt.show()

        for p in pts:
            lbl = data[p]
            # surr_lbls = []
            # for q in range(self.nghood):
            #     nghbInd = self.neighborsM[q, p]
            #     if np.isnan(nghbInd):
            #         continue
            #     else:
            #         nghbInd = int(nghbInd)
            #         surr_lbls.append(data[nghbInd])
            surr_lbls = [data[int(self.neighborsM[q, p])] for q in range(self.nghood)
                         if not np.isnan(self.neighborsM[q, p])]
            enemies = [x != lbl for x in surr_lbls]
            n_enemies = np.array(enemies).sum()
            if n_enemies > 0:
                pass
            if n_enemies >= (self.enemies_T * self.nghood):
                hist, bins = skiexp.histogram(np.array(surr_lbls))
                enem_lbl = bins[np.argmax(hist)]
                data[p] = enem_lbl

        return data

    def make_neighborhood_matrix(self):
        # print 'start'
        if self.gui:
            self.gui.statusbar.config(text='Creating neighborhood matrix...')
            self.gui.progressbar.set(0, 'Creating neighborhood matrix...')
        if self.nghood == 8:
            nr = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
            nc = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
            ns = np.zeros(self.nghood)
        elif self.nghood == 4:
            nr = np.array([-1, 0, 0, 1])
            nc = np.array([0, -1, 1, 0])
            ns = np.zeros(self.nghood, dtype=np.int32)
        elif self.nghood == 26:
            nrCenter = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
            ncCenter = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
            nrBorder = np.zeros([-1, -1, -1, 0, 0, 0, 1, 1, 1])
            ncBorder = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
            nr = np.array(np.hstack((nrBorder, nrCenter, nrBorder)))
            nc = np.array(np.hstack((ncBorder, ncCenter, ncBorder)))
            ns = np.array(np.hstack((-np.ones_like(nrBorder), np.zeros_like(nrCenter), np.ones_like(nrBorder))))
        elif self.nghood == 6:
            nrCenter = np.array([-1, 0, 0, 1])
            ncCenter = np.array([0, -1, 1, 0])
            nrBorder = np.array([0])
            ncBorder = np.array([0])
            nr = np.array(np.hstack((nrBorder, nrCenter, nrBorder)))
            nc = np.array(np.hstack((ncBorder, ncCenter, ncBorder)))
            ns = np.array(np.hstack((-np.ones_like(nrBorder), np.zeros_like(nrCenter), np.ones_like(nrBorder))))
        else:
            print 'Wrong neighborhood passed. Exiting.'
            return None

        neighborsM = np.zeros((self.nghood, self.nvoxels))
        for i in range(self.nvoxels ):
            s, r, c =  tuple(self.coordsv[:, i])
            for nghb in range(self.nghood):
                rn = r + nr[nghb]
                cn = c + nc[nghb]
                sn = s + ns[nghb]
                if rn < 0 or rn > (self.nrows-1) or cn < 0 or cn > (self.ncols-1) or sn < 0 or sn > (self.nslices-1):
                    neighborsM[nghb, i] = np.NaN
                else:
                    indexN = np.ravel_multi_index( (sn, rn, cn), self.data.shape )
                    neighborsM[nghb, i] = indexN
            if self.gui and np.floor(np.mod( i, self.nvoxels/20 )) == 0:
                self.gui.progressbar.step(1./20)
                self.gui.canvas.draw()
        if self.gui:
            self.gui.progressbar.set(0, '')
            self.gui.statusbar.config(text='Neighborhood matrix created')
        return neighborsM


################################################################################
################################################################################
if __name__ == '__main__':
    # data_fname = '/home/tomas/Data/medical/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    # data, mask, voxel_size = tools.load_pickle_data(data_fname)
    #
    # slice_ind = 17
    # data_s = data[slice_ind, :, :]
    # img = tools.windowing(data_s)
    # # img = skidat.camera()
    # # # img = skicol.rgb2gray(im)
    # # img = skitra.rescale(img, 0.2, preserve_range=True)
    #
    # main_cl, main_rv = tools.dominant_class(img, peakT=0.6, dens_min=0, dens_max=255, show=True, show_now=False)
    #
    # # plt.figure()
    # # plt.subplot(131), plt.imshow(img, 'gray', interpolation='nearest')
    # # plt.subplot(132), plt.imshow(main_rv.pdf(img), 'gray', interpolation='nearest')
    # # plt.subplot(133), plt.imshow(main_cl, 'gray', interpolation='nearest')
    # # plt.show()
    #
    # peak = main_rv.mean()
    # seed_t = 100
    # seeds1 = main_cl
    # seeds1 = skimor.binary_opening(seeds1, selem=skimor.disk(2))
    # seeds2 = np.abs(img - peak) > seed_t
    # seeds = seeds1 + 2 * seeds2

    # data_fname = '/home/tomas/Data/medical/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    #
    # slice_ind = 17
    # data_s = data[slice_ind, :, :]
    # img = tools.windowing(data_s)
    # plt.figure()
    # plt.imshow(img, 'gray')
    # plt.show()

    # data_fname = '/home/tomas/Dropbox/Work/Dizertace/figures/liver_segmentation/FNPL_46324212_191787340.DCM'
    # img, meta = io3d.read(data_fname)
    # img = tools.windowing(img[0,...])
    # cv2.imwrite('/home/tomas/Dropbox/Work/Dizertace/figures/liver_segmentation/input.png', img)
    data_fname = '/home/tomas/Dropbox/Work/Dizertace/figures/liver_segmentation/input.png'
    img = cv2.imread(data_fname, 0)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    # seeds, peaks = tools.seeds_from_hist(img, min_int=1, max_int=254, show=True, show_now=False)
    # seeds, peaks = tools.seeds_from_glcm(img, show=True, show_now=False)
    # seeds, peaks = tools.seeds_from_glcm_mesh(img, show=True, show_now=False)
    seeds, peaks = tools.seeds_from_glcm_meanshift(img, show=True, show_now=False)

    # plt.figure()
    # plt.subplot(121), plt.imshow(img, 'gray', interpolation='nearest')
    # plt.subplot(122), plt.imshow(seeds, 'jet', interpolation='nearest')
    # plt.show()

    # gc = GrowCut(img, seeds, maxits=100, enemies_T=0.7, smooth_cell=True)
    gc = GrowCut(img, seeds, maxits=100, enemies_T=0.7, smooth_cell=False)
    gc.run()
    labs = gc.get_labeled_im()

    # ilb.score_data(img, labs[0,...])

    seg = skiseg.clear_border(labs[0,...])
    plt.figure()
    plt.subplot(131), plt.imshow(img, 'gray', interpolation='nearest'), plt.title('input'), plt.axis('off')
    plt.subplot(132), plt.imshow(seeds, 'jet', interpolation='nearest'), plt.title('seeds'), plt.axis('off')
    # divider = make_axes_locatable(plt.gca())
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # plt.colorbar(cax=cax, ticks=np.unique(seeds))
    plt.subplot(133), plt.imshow(seg, 'jet', interpolation='nearest', vmin=0), plt.title('segmentation'), plt.axis('off')
    # divider = make_axes_locatable(plt.gca())
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # plt.colorbar(cax=cax, ticks=np.unique(seeds))

    plt.show()