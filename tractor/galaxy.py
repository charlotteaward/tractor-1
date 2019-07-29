"""
This file is part of the Tractor project.
Copyright 2011, 2012, 2013 Dustin Lang and David W. Hogg.
Licensed under the GPLv2; see the file COPYING for details.

`galaxy.py`
================

Exponential and deVaucouleurs galaxy model classes.

These use the slightly modified versions of the exp and dev profiles
from the SDSS /Photo/ software; we use multi-Gaussian approximations
of these.
"""
from __future__ import print_function
from __future__ import division

import numpy as np

from astrometry.util.miscutils import get_overlapping_region

from tractor import mixture_profiles as mp
from tractor.utils import ParamList, MultiParams, ScalarParam, BaseParams
from tractor.patch import Patch, add_patches, add_patches2, ModelMask
from tractor.basics import SingleProfileSource, BasicSource, PointSource

debug_ps = None


def get_galaxy_cache():
    return None


def set_galaxy_cache_size(N=10000):
    # Ugh, dealing with caching + extents / modelMasks was too much
    pass


enable_galaxy_cache = set_galaxy_cache_size


def disable_galaxy_cache():
    pass


class GalaxyShape(ParamList):
    '''
    A naive representation of an ellipse (describing a galaxy shape),
    using effective radius (in arcsec), axis ratio, and position angle.

    For better ellipse parameterizations, see ellipses.py
    '''
    @staticmethod
    def getName():
        return "Galaxy Shape"

    @staticmethod
    def getNamedParams():
        '''
        re: arcsec
        ab: axis ratio, dimensionless, in [0,1]
        phi: deg, "E of N", 0=direction of increasing Dec,
        90=direction of increasing RA
        '''
        return dict(re=0, ab=1, phi=2)

    def __repr__(self):
        return 're=%g, ab=%g, phi=%g' % (self.re, self.ab, self.phi)

    def __str__(self):
        return ('%s: re=%.2f, ab=%.2f, phi=%.1f' %
                (self.getName(), self.re, self.ab, self.phi))

    def getTensor(self, cd):
        # G takes unit vectors (in r_e) to degrees (~intermediate world coords)
        G = self.getRaDecBasis()
        # "cd" takes pixels to degrees (intermediate world coords)
        # T takes pixels to unit vectors.
        T = np.dot(np.linalg.inv(G), cd)
        return T

    def getRaDecBasis(self):
        '''
        Returns a transformation matrix that takes vectors in r_e
        to delta-RA, delta-Dec vectors.
        '''
        # # convert re, ab, phi into a transformation matrix
        phi = np.deg2rad(90 - self.phi)
        # # convert re to degrees
        # # HACK -- bring up to a minimum size to prevent singular
        # # matrix inversions
        re_deg = max(1. / 30, self.re) / 3600.
        cp = np.cos(phi)
        sp = np.sin(phi)
        # Squish, rotate, and scale into degrees.
        # resulting G takes unit vectors (in r_e) to degrees
        # (~intermediate world coords)
        return re_deg * np.array([[cp, sp * self.ab], [-sp, cp * self.ab]])


class Galaxy(MultiParams, SingleProfileSource):
    '''
    Generic Galaxy profile with position, brightness, and shape.
    '''

    def __init__(self, *args):
        super(Galaxy, self).__init__(*args)
        self.name = self.getName()
        self.dname = self.getDName()

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightness=1, shape=2)

    def getName(self):
        return 'Galaxy'

    def getDName(self):
        '''
        Name used in labeling the derivative images d(Dname)/dx, eg
        '''
        return 'gal'

    def getSourceType(self):
        return self.name

    def getShape(self):
        return self.shape

    def __str__(self):
        return (self.name + ' at ' + str(self.pos)
                + ' with ' + str(self.brightness)
                + ' and ' + str(self.shape))

    def __repr__(self):
        return (self.name + '(pos=' + repr(self.pos) +
                ', brightness=' + repr(self.brightness) +
                ', shape=' + repr(self.shape) + ')')

    def getUnitFluxModelPatch(self, img, **kwargs):
        raise RuntimeError('getUnitFluxModelPatch unimplemented in' +
                           self.getName())

    # returns [ Patch, Patch, ... ] of length numberOfParams().
    # Galaxy.
    def getParamDerivatives(self, img, modelMask=None):
        pos0 = self.getPosition()
        (px0, py0) = img.getWcs().positionToPixel(pos0, self)
        counts = img.getPhotoCal().brightnessToCounts(self.brightness)

        minsb = img.modelMinval
        if counts > 0:
            minval = minsb / counts
        else:
            minval = None

        patch0 = self.getUnitFluxModelPatch(img, px=px0, py=py0, minval=minval,
                                            modelMask=modelMask)
        if patch0 is None:
            return [None] * self.numberOfParams()
        derivs = []

        if modelMask is None:
            modelMask = ModelMask.fromExtent(*patch0.getExtent())
        assert(modelMask is not None)

        # FIXME -- would we be better to do central differences in
        # pixel space, and convert to Position via CD matrix?

        # derivatives wrt position
        psteps = pos0.getStepSizes()
        
        if not self.isParamFrozen('pos'):
            params = pos0.getParams()
            if counts == 0:
                derivs.extend([None] * len(params))
                psteps = []
            for i, pstep in enumerate(psteps):
                pstep=pstep
                #print(params[i],params[i] + pstep)
                oldval = pos0.setParam(i, params[i] + pstep)
                (px, py) = img.getWcs().positionToPixel(pos0, self)
                pos0.setParam(i, oldval)
                
                patchx = self.getUnitFluxModelPatch(
                    img, px=px, py=py, minval=minval, modelMask=modelMask)
                if patchx is None or patchx.getImage() is None:
                    derivs.append(None)
                    print('patchx was none')
                    continue
                dx = (patchx - patch0) * (counts / pstep)
                dx.setName('d(%s)/d(pos%i)' % (self.dname, i)) 
                derivs.append(dx)

        # derivatives wrt brightness
        bsteps = self.brightness.getStepSizes()
        if not self.isParamFrozen('brightness'):
            params = self.brightness.getParams()
            for i, bstep in enumerate(bsteps):
                oldval = self.brightness.setParam(i, params[i] + bstep)
                countsi = img.getPhotoCal().brightnessToCounts(self.brightness)
                self.brightness.setParam(i, oldval)
                df = patch0 * ((countsi - counts) / bstep)
                df.setName('d(%s)/d(bright%i)' % (self.dname, i))
                derivs.append(df)

        # derivatives wrt shape
        gsteps = self.shape.getStepSizes()
        if not self.isParamFrozen('shape'):
            gnames = self.shape.getParamNames()
            oldvals = self.shape.getParams()
            if counts == 0:
                derivs.extend([None] * len(oldvals))
                gsteps = []
            for i, gstep in enumerate(gsteps):
                oldval = self.shape.setParam(i, oldvals[i] + gstep)
                patchx = self.getUnitFluxModelPatch(
                    img, px=px0, py=py0, minval=minval, modelMask=modelMask)
                self.shape.setParam(i, oldval)
                if patchx is None:
                    print('patchx is None:')
                    print('  ', self)
                    print('  stepping galaxy shape',
                          self.shape.getParamNames()[i])
                    print('  stepped', gsteps[i])
                    print('  to', self.shape.getParams()[i])
                    derivs.append(None)
                    continue
                dx = (patchx - patch0) * (counts / gstep)
                dx.setName('d(%s)/d(%s)' % (self.dname, gnames[i]))
                derivs.append(dx)
        return derivs


class ProfileGalaxy(object):
    '''
    A mix-in class that renders itself based on a Mixture-of-Gaussians
    profile.
    '''

    def getName(self):
        return 'ProfileGalaxy'

    def getProfile(self):
        return None

    # Here's the main method to override;
    def _getAffineProfile(self, img, px, py):
        ''' Returns a MixtureOfGaussians profile that has been
        affine-transformed into the pixel space of the image.
        '''
        return None

    def _getUnitFluxDeps(self, img, px, py):
        return None

    def _getUnitFluxPatchSize(self, img, **kwargs):
        return 0

    def getUnitFluxModelPatch(self, img, px=None, py=None, minval=0.0,
                              modelMask=None):
        if px is None or py is None:
            (px, py) = img.getWcs().positionToPixel(self.getPosition(), self)
        patch = self._realGetUnitFluxModelPatch(
            img, px, py, minval, modelMask=modelMask)
        if patch is not None and modelMask is not None:
            assert(patch.shape == modelMask.shape)
        return patch

    def _realGetUnitFluxModelPatch(self, img, px, py, minval, modelMask=None):
        if modelMask is not None:
            x0, y0 = modelMask.x0, modelMask.y0
        else:
            # choose the patch size
            halfsize = self._getUnitFluxPatchSize(img, px=px, py=py, minval=minval)
            # find overlapping pixels to render
            (outx, inx) = get_overlapping_region(
                int(np.floor(px - halfsize)), int(np.ceil(px + halfsize + 1)),
                0, img.getWidth())
            (outy, iny) = get_overlapping_region(
                int(np.floor(py - halfsize)), int(np.ceil(py + halfsize + 1)),
                0, img.getHeight())
            if inx == [] or iny == []:
                # no overlap
                return None
            x0, x1 = outx.start, outx.stop
            y0, y1 = outy.start, outy.stop

        psf = img.getPsf()

        # We have two methods of rendering profile galaxies: If the
        # PSF can be represented as a mixture of Gaussians, then we do
        # the analytic Gaussian convolution, producing a larger
        # mixture of Gaussians, and we render that.  Otherwise
        # (pixelized PSFs), we FFT the PSF, multiply by the analytic
        # FFT of the galaxy, and IFFT back to get the rendered
        # profile.

        # The "HybridPSF" class is just a marker to indicate whether this
        # code should treat the PSF as a hybrid.
        from tractor.psf import HybridPSF
        hybrid = isinstance(psf, HybridPSF)

        def run_mog(amix=None, mm=None):
            ''' This runs the mixture-of-Gaussians convolution method.
            '''
            if amix is None:
                amix = self._getAffineProfile(img, px, py)
            if mm is None:
                mm = modelMask
            # now convolve with the PSF, analytically
            # (note that the psf's center is *not* set to px,py; that's just
            #  the position to use for spatially-varying PSFs)
            psfmix = psf.getMixtureOfGaussians(px=px, py=py)
            cmix = amix.convolve(psfmix)
            if mm is None:
                #print('Mixture to patch: amix', amix, 'psfmix', psfmix, 'cmix', cmix)
                return mp.mixture_to_patch(cmix, x0, x1, y0, y1, minval)
            # The convolved mixture *already* has the px,py offset added
            # (via px,py to amix) so set px,py=0,0 in this call.
            if mm.mask is not None:
                p = cmix.evaluate_grid_masked(mm.x0, mm.y0, mm.mask, 0., 0.)
            else:
                p = cmix.evaluate_grid(mm.x0, mm.x1, mm.y0, mm.y1, 0., 0.)
            assert(p.shape == mm.shape)
            return p

        if hasattr(psf, 'getMixtureOfGaussians') and not hybrid:
            return run_mog(mm=modelMask)

        # Otherwise, FFT:
        imh, imw = img.shape
        if modelMask is None:
            # Avoid huge galaxies -> huge halfsize in a tiny image (blob)
            imsz = max(imh, imw)
            halfsize = min(halfsize, imsz)
            # FIXME -- should take some kind of combination of
            # modelMask, PSF, and Galaxy sizes!

        else:
            # ModelMask sets the sizes.
            mh, mw = modelMask.shape
            x1 = x0 + mw
            y1 = y0 + mh

            halfsize = max(mh / 2., mw / 2.)
            # How far from the source center to furthest modelMask edge?
            # FIXME -- add 1 for Lanczos margin?
            halfsize = max(halfsize, max(max(1 + px - x0, 1 + x1 - px),
                                         max(1 + py - y0, 1 + y1 - py)))
            psfh, psfw = psf.shape
            halfsize = max(halfsize, max(psfw / 2., psfh / 2.))
            #print('Halfsize:', halfsize)

            # is the source center outside the modelMask?
            sourceOut = (px < x0 or px > x1 - 1 or py < y0 or py > y1 - 1)
            # print('mh,mw', mh,mw, 'sourceout?', sourceOut)

            if sourceOut:
                if hybrid:
                    return run_mog(mm=modelMask)

                # Super Yuck -- FFT, modelMask, source is outside the
                # box.
                neardx, neardy = 0., 0.
                if px < x0:
                    neardx = x0 - px
                if px > x1:
                    neardx = px - x1
                if py < y0:
                    neardy = y0 - py
                if py > y1:
                    neardy = py - y1
                nearest = np.hypot(neardx, neardy)
                #print('Nearest corner:', nearest, 'vs radius', self.getRadius())
                if nearest > self.getRadius():
                    return None
                # how far is the furthest point from the source center?
                farw = max(abs(x0 - px), abs(x1 - px))
                farh = max(abs(y0 - py), abs(y1 - py))
                bigx0 = int(np.floor(px - farw))
                bigx1 = int(np.ceil(px + farw))
                bigy0 = int(np.floor(py - farh))
                bigy1 = int(np.ceil(py + farh))
                bigw = 1 + bigx1 - bigx0
                bigh = 1 + bigy1 - bigy0
                boffx = x0 - bigx0
                boffy = y0 - bigy0
                assert(bigw >= mw)
                assert(bigh >= mh)
                assert(boffx >= 0)
                assert(boffy >= 0)
                bigMask = np.zeros((bigh, bigw), bool)
                if modelMask.mask is not None:
                    bigMask[boffy:boffy + mh,
                            boffx:boffx + mw] = modelMask.mask
                else:
                    bigMask[boffy:boffy + mh, boffx:boffx + mw] = True
                bigMask = ModelMask(bigx0, bigy0, bigMask)
                # print('Recursing:', self, ':', (mh,mw), 'to', (bigh,bigw))
                bigmodel = self._realGetUnitFluxModelPatch(
                    img, px, py, minval, modelMask=bigMask)
                return Patch(x0, y0,
                             bigmodel.patch[boffy:boffy + mh, boffx:boffx + mw])

        # print('Getting Fourier transform of PSF at', px,py)
        # print('Tim shape:', img.shape)
        P, (cx, cy), (pH, pW), (v, w) = psf.getFourierTransform(px, py, halfsize)

        dx = px - cx
        dy = py - cy
        if modelMask is not None:
            # the Patch we return *must* have this origin.
            ix0 = x0
            iy0 = y0
            # the difference that we have to handle by shifting the model image
            mux = dx - ix0
            muy = dy - iy0
            # we will handle the integer portion by computing a shifted image
            # and copying it into the result
            sx = int(np.round(mux))
            sy = int(np.round(muy))
            # the subpixel portion will be handled with a Lanczos interpolation
            mux -= sx
            muy -= sy
        else:
            # Put the integer portion of the offset into Patch x0,y0
            ix0 = int(np.round(dx))
            iy0 = int(np.round(dy))
            # the subpixel portion will be handled with a Lanczos interpolation
            mux = dx - ix0
            muy = dy - iy0

        # At this point, mux,muy are both in [-0.5, 0.5]
        assert(np.abs(mux) <= 0.5)
        assert(np.abs(muy) <= 0.5)

        amix = self._getShearedProfile(img, px, py)
        fftmix = amix
        mogmix = None

        #print('Galaxy affine profile:', amix)

        if hybrid:
            # Split "amix" into terms that we will evaluate using MoG
            # vs FFT.
            vv = amix.var[:, 0, 0] + amix.var[:, 1, 1]
            nsigma = 3.
            # Terms that will wrap-around significantly...
            I = ((vv * nsigma**2) > (pW**2))
            if np.any(I):
                # print('Evaluating', np.sum(I), 'terms as MoGs')
                mogmix = mp.MixtureOfGaussians(amix.amp[I],
                                               amix.mean[I, :] + np.array([px, py])[np.newaxis, :],
                                               amix.var[I, :, :], quick=True)
            I = np.logical_not(I)
            if np.any(I):
                # print('Evaluating', np.sum(I), 'terms with FFT')
                fftmix = mp.MixtureOfGaussians(amix.amp[I], amix.mean[I, :], amix.var[I, :, :],
                                                quick=True)
            else:
                fftmix = None

        if fftmix is not None:
            Fsum = fftmix.getFourierTransform(v, w, zero_mean=True)
            # In Intel's mkl_fft library, the irfftn code path is faster than irfft2
            # (the irfft2 version sets args (to their default values) which triggers padding
            #  behavior, changing the FFT size and copying behavior)
            #G = np.fft.irfft2(Fsum * P, s=(pH, pW))
            G = np.fft.irfftn(Fsum * P)

            assert(G.shape == (pH,pW))
            # FIXME -- we could try to be sneaky and Lanczos-interp
            # after cutting G down to nearly its final size... tricky
            # tho

            # Lanczos-3 interpolation in ~the same way we do for
            # pixelized PSFs.
            from tractor.psf import lanczos_shift_image
            G = G.astype(np.float32)
            lanczos_shift_image(G, mux, muy, inplace=True)
        else:
            G = np.zeros((pH, pW), np.float32)

        if modelMask is not None:
            gh, gw = G.shape
            if sx != 0 or sy != 0:
                yi, yo = get_overlapping_region(-sy, -sy + mh - 1, 0, gh - 1)
                xi, xo = get_overlapping_region(-sx, -sx + mw - 1, 0, gw - 1)
                # shifted
                # FIXME -- are yo,xo always the whole image?  If so, optimize
                shG = np.zeros((mh, mw), G.dtype)
                shG[yo, xo] = G[yi, xi]

                if debug_ps is not None:
                    _fourier_galaxy_debug_plots(G, shG, xi, yi, xo, yo, P, Fsum,
                                                pW, pH, psf)

                G = shG
            if gh > mh or gw > mw:
                G = G[:mh, :mw]
            assert(G.shape == modelMask.shape)

        else:
            # Clip down to suggested "halfsize"
            if x0 > ix0:
                G = G[:, x0 - ix0:]
                ix0 = x0
            if y0 > iy0:
                G = G[y0 - iy0:, :]
                iy0 = y0
            gh, gw = G.shape
            if gw + ix0 > x1:
                G = G[:, :x1 - ix0]
            if gh + iy0 > y1:
                G = G[:y1 - iy0, :]

        if mogmix is not None:
            if modelMask is not None:
                mogpatch = run_mog(amix=mogmix, mm=modelMask)
            else:
                gh, gw = G.shape
                mogpatch = run_mog(amix=mogmix, mm=ModelMask(ix0, iy0, gw, gh))
            assert(mogpatch.patch.shape == G.shape)
            G += mogpatch.patch

        return Patch(ix0, iy0, G)

def _fourier_galaxy_debug_plots(G, shG, xi, yi, xo, yo, P, Fsum,
                                pW, pH, psf):
    import pylab as plt
    mx = G.max()
    ima = dict(vmin=np.log10(mx) - 6,
               vmax=np.log10(mx),
               interpolation='nearest', origin='lower')
    plt.clf()
    plt.subplot(1, 2, 1)
    #plt.imshow(shG, interpolation='nearest', origin='lower')
    plt.imshow(np.log10(shG), **ima)
    ax = plt.axis()
    plt.plot([xo.start, xo.start, xo.stop - 1, xo.stop - 1, xo.start],
             [yo.start, yo.stop - 1, yo.stop - 1, yo.start, yo.start],
             'r-')
    plt.axis(ax)
    plt.title('shG')
    plt.subplot(1, 2, 2)
    #plt.imshow(G, interpolation='nearest', origin='lower')
    plt.imshow(np.log10(G), **ima)
    ax = plt.axis()
    plt.plot([xi.start, xi.start, xi.stop - 1, xi.stop - 1, xi.start],
             [yi.start, yi.stop - 1, yi.stop - 1, yi.start, yi.start],
             'r-')
    plt.axis(ax)
    plt.title('G')
    debug_ps.savefig()

    def plot_real_imag(F, name):
        plt.clf()
        plt.subplot(2, 2, 1)
        print(name, 'real range', F.real.min(), F.real.max())
        plt.imshow(np.log10(np.abs(F.real)),
                   interpolation='nearest', origin='lower')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title('log(abs(%s.real))' % name)
        plt.subplot(2, 2, 3)
        plt.imshow(np.sign(F.real),
                   vmin=-1, vmax=1,
                   interpolation='nearest', origin='lower', cmap='RdBu')
        plt.xticks([])
        plt.yticks([])
        plt.title('sign(%s.real)' % name)
        print(name, 'imag range', F.imag.min(), F.imag.max())
        plt.subplot(2, 2, 2)
        mx = np.abs(F.imag).max()
        plt.imshow(np.log10(np.abs(F.imag)),
                   interpolation='nearest', origin='lower')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title('log(abs(%s.imag))' % name)
        plt.subplot(2, 2, 4)
        plt.imshow(np.sign(F.imag),
                   vmin=-1, vmax=1,
                   interpolation='nearest', origin='lower', cmap='RdBu')
        plt.xticks([])
        plt.yticks([])
        plt.title('sign(%s.imag)' % name)

    plot_real_imag(P, 'PSF')
    debug_ps.savefig()
    plot_real_imag(Fsum, 'FFT(Galaxy)')
    debug_ps.savefig()
    plot_real_imag(P * Fsum, 'FFT(PSF * Galaxy)')
    debug_ps.savefig()

    plt.clf()
    p = np.fft.irfft2(P, s=(pH, pW))
    ax = plt.axis([pW // 2 - 7, pW // 2 + 7, pH // 2 - 7, pH // 2 + 7])
    plt.subplot(1, 3, 1)
    plt.imshow(p, interpolation='nearest', origin='lower')
    plt.axis(ax)
    plt.title('psf (real space)')
    # This is in the corners...
    g = np.fft.irfft2(Fsum, s=(pH, pW))
    plt.subplot(1, 3, 2)
    plt.imshow(g, interpolation='nearest', origin='lower')
    plt.title('galaxy (real space)')
    c = np.fft.irfft2(Fsum * P, s=(pH, pW))
    plt.subplot(1, 3, 3)
    plt.imshow(c, interpolation='nearest', origin='lower')
    plt.axis(ax)
    plt.title('convolved (real space)')
    debug_ps.savefig()

    # # What kind of artifacts do we get from the iFFT(FFT(PSF)) - PSF ?
    # p = np.fft.irfft2(P, s=(pH,pW))
    # plt.clf()
    # plt.subplot(1,3,1)
    # # ASSUME PixelizedPSF
    # pad,cx,cy = psf._padInImage(pW,pH)
    # plt.imshow(pad, interpolation='nearest', origin='lower')
    # plt.subplot(1,3,2)
    # plt.imshow(p, interpolation='nearest', origin='lower')
    # plt.subplot(1,3,3)
    # plt.imshow(pad - p, interpolation='nearest', origin='lower')
    # plt.colorbar()
    # debug_ps.savefig()

    # plt.clf()
    # plt.subplot(1,2,1)
    # mx = np.abs(P).max()
    # plt.imshow(np.log10(np.abs(P)),
    #            interpolation='nearest', origin='lower')
    # plt.xticks([]); plt.yticks([])
    # plt.colorbar()
    # plt.title('log(abs(FFT(PSF)))')
    # plt.subplot(1,2,2)
    # plt.imshow(np.arctan2(P.imag, P.real),
    #            vmin=-np.pi, vmax=np.pi,
    #            interpolation='nearest', origin='lower', cmap='RdBu')
    # plt.xticks([]); plt.yticks([])
    # plt.title('phase(FFT(PSF))')
    # debug_ps.savefig()


class HoggGalaxy(ProfileGalaxy, Galaxy):

    def getName(self):
        return 'HoggGalaxy'

    def getRadius(self):
        return self.nre * self.shape.re

    def _getAffineProfile(self, img, px, py):
        ''' Returns a MixtureOfGaussians profile that has been
        affine-transformed into the pixel space of the image.
        '''
        galmix = self.getProfile()
        cdinv = img.getWcs().cdInverseAtPixel(px, py)
        G = self.shape.getRaDecBasis()
        Tinv = np.dot(cdinv, G)
        amix = galmix.apply_affine(np.array([px, py]), Tinv)
        return amix

    def _getShearedProfile(self, img, px, py):
        ''' Returns a MixtureOfGaussians profile that has been
        shear-transformed into the pixel space of the image.
        At px,py (but not offset to px,py).
        '''
        galmix = self.getProfile()
        cdinv = img.getWcs().cdInverseAtPixel(px, py)
        G = self.shape.getRaDecBasis()
        Tinv = np.dot(cdinv, G)
        amix = galmix.apply_shear(Tinv)
        return amix

    def _getUnitFluxDeps(self, img, px, py):
        # return ('unitpatch', self.getName(), px, py,
        return hash(('unitpatch', self.getName(), px, py,
                     img.getWcs().hashkey(),
                     img.getPsf().hashkey(), self.shape.hashkey())
                    )

    def _getUnitFluxPatchSize(self, img, px=0., py=0., minval=0.):
        if hasattr(self, 'halfsize'):
            return self.halfsize
        pixscale = img.wcs.pixscale_at(px, py)
        halfsize = max(1., self.getRadius() / pixscale)
        halfsize += img.psf.getRadius()
        halfsize = int(np.ceil(halfsize))
        return halfsize


class GaussianGalaxy(HoggGalaxy):
    nre = 6.
    profile = mp.MixtureOfGaussians(np.array([1.]), np.zeros((1, 2)),
                                    np.array([[[1., 0.], [0., 1.]]]))
    profile.normalize()

    def __init__(self, *args, **kwargs):
        self.nre = GaussianGalaxy.nre
        super(GaussianGalaxy, self).__init__(*args, **kwargs)

    def getName(self):
        return 'GaussianGalaxy'

    def getProfile(self):
        return GaussianGalaxy.profile


class ExpGalaxy(HoggGalaxy):
    nre = 4.
    profile = mp.get_exp_mixture()
    profile.normalize()

    @staticmethod
    def getExpProfile():
        return ExpGalaxy.profile

    def __init__(self, *args, **kwargs):
        self.nre = ExpGalaxy.nre
        super(ExpGalaxy, self).__init__(*args, **kwargs)

    def getName(self):
        return 'ExpGalaxy'

    def getProfile(self):
        return ExpGalaxy.profile


class DevGalaxy(HoggGalaxy):
    nre = 8.
    profile = mp.get_dev_mixture()
    profile.normalize()

    @staticmethod
    def getDevProfile():
        return DevGalaxy.profile

    def __init__(self, *args, **kwargs):
        self.nre = DevGalaxy.nre
        super(DevGalaxy, self).__init__(*args, **kwargs)

    def getName(self):
        return 'DevGalaxy'

    def getProfile(self):
        return DevGalaxy.profile


class FracDev(ScalarParam):
    stepsize = 0.01

    def clipped(self):
        f = self.getValue()
        return np.clip(f, 0., 1.)

    def derivative(self):
        return 1.


class SoftenedFracDev(FracDev):
    '''
    Implements a "softened" version of the deV-to-total fraction.

    The sigmoid function is scaled and shifted so that S(0) ~ 0.1 and
    S(1) ~ 0.9.

    Use the 'clipped()' function to get the un-softened fracDev
    clipped to [0,1].
    '''

    def clipped(self):
        f = self.getValue()
        return 1. / (1 + np.exp(4. * (0.5 - f)))

    def derivative(self):
        f = self.getValue()
        # Thanks, Sage
        ef = np.exp(-4. * f + 2)
        return 4. * ef / ((ef + 1)**2)


class FixedCompositeGalaxy(MultiParams, ProfileGalaxy, SingleProfileSource):
    '''
    A galaxy with Exponential and deVaucouleurs components where the
    brightnesses of the deV and exp components are defined in terms of
    a total brightness and a fraction of that total that goes to the
    deV component.

    The two components share a position (ie the centers are the same),
    but have different shapes.  The galaxy has a single brightness
    that is split between the components.

    This is like CompositeGalaxy, but more useful for getting
    consistent colors from forced photometry, because one can freeze
    the fracDev to keep the deV+exp profile fixed.
    '''

    def __init__(self, pos, brightness, fracDev, shapeExp, shapeDev):
        # handle passing fracDev as a float
        if not isinstance(fracDev, BaseParams):
            fracDev = FracDev(fracDev)
        MultiParams.__init__(self, pos, brightness, fracDev,
                             shapeExp, shapeDev)
        self.name = self.getName()

    def getRadius(self):
        return max(self.shapeExp.re * ExpGalaxy.nre,
                   self.shapeDev.re * DevGalaxy.nre)

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightness=1, fracDev=2, shapeExp=3, shapeDev=4)

    def getName(self):
        return 'FixedCompositeGalaxy'

    def __str__(self):
        return (self.name + ' at ' + str(self.pos)
                + ' with ' + str(self.brightness)
                + ', ' + str(self.fracDev)
                + ', exp ' + str(self.shapeExp)
                + ', deV ' + str(self.shapeDev))

    def __repr__(self):
        return (self.name + '(pos=' + repr(self.pos) +
                ', brightness=' + repr(self.brightness) +
                ', fracDev=' + repr(self.fracDev) +
                ', shapeExp=' + repr(self.shapeExp) +
                ', shapeDev=' + repr(self.shapeDev) + ')')

    def _getAffineProfile(self, img, px, py):
        # f = self.fracDev.clipped()
        # profs = []
        # if f > 0.:
        #     profs.append((f, DevGalaxy.profile, self.shapeDev))
        # if f < 1.:
        #     profs.append((1. - f, ExpGalaxy.profile, self.shapeExp))
        # cdinv = img.getWcs().cdInverseAtPixel(px, py)
        # mix = []
        # for f, p, s in profs:
        #     G = s.getRaDecBasis()
        #     Tinv = np.dot(cdinv, G)
        #     amix = p.apply_affine(np.array([px, py]), Tinv)
        #     amix.amp = amix.amp * f
        #     mix.append(amix)

        f = self.fracDev.clipped()
        cdinv = img.getWcs().cdInverseAtPixel(px, py)
        mix = []
        if f > 0.:
            amix = DevGalaxy.profile.apply_affine(np.array([px, py]), np.dot(cdinv, self.shapeDev.getRaDecBasis()))
            amix.amp = amix.amp * f
            mix.append(amix)
        if f < 1.:
            amix = ExpGalaxy.profile.apply_affine(np.array([px, py]), np.dot(cdinv, self.shapeExp.getRaDecBasis()))
            amix.amp = amix.amp * (1. - f)
            mix.append(amix)
        if len(mix) == 1:
            return mix[0]
        return mix[0] + mix[1]

    def _getShearedProfile(self, img, px, py):
        ''' Returns a MixtureOfGaussians profile that has been
        shear-transformed into the pixel space of the image.
        At px,py (but not offset to px,py).
        '''
        # f = self.fracDev.clipped()
        # profs = []
        # if f > 0.:
        #     profs.append((f, DevGalaxy.profile, self.shapeDev))
        # if f < 1.:
        #     profs.append((1. - f, ExpGalaxy.profile, self.shapeExp))
        # cdinv = img.getWcs().cdInverseAtPixel(px, py)
        # mix = []
        # for f, p, s in profs:
        #     G = s.getRaDecBasis()
        #     Tinv = np.dot(cdinv, G)
        #     amix = p.apply_shear(Tinv)
        #     amix.amp = amix.amp * f
        #     mix.append(amix)
        # if len(mix) == 1:
        #     return mix[0]
        # return mix[0] + mix[1]

        f = self.fracDev.clipped()
        cdinv = img.getWcs().cdInverseAtPixel(px, py)
        mix = []
        if f > 0.:
            amix = DevGalaxy.profile.apply_shear(np.dot(cdinv, self.shapeDev.getRaDecBasis()))
            amix.amp = amix.amp * f
            mix.append(amix)
        if f < 1.:
            amix = ExpGalaxy.profile.apply_shear(np.dot(cdinv, self.shapeExp.getRaDecBasis()))
            amix.amp = amix.amp * (1. - f)
            mix.append(amix)
        if len(mix) == 1:
            return mix[0]
        return mix[0] + mix[1]

    def _getUnitFluxPatchSize(self, img, px=0., py=0., minval=0.):
        if hasattr(self, 'halfsize'):
            return self.halfsize
        pixscale = img.wcs.pixscale_at(px, py)
        f = self.fracDev.clipped()
        r = 1.
        if f < 1.:
            s = self.shapeExp
            rexp = ExpGalaxy.nre * s.re
            r = max(r, rexp)
        if f > 0.:
            s = self.shapeDev
            rdev = max(r, DevGalaxy.nre * s.re)
            r = max(r, rdev)
        halfsize = r / pixscale
        halfsize += img.psf.getRadius()
        return halfsize

    def _getUnitFluxDeps(self, img, px, py):
        return hash(('unitpatch', self.getName(),
                     px, py, img.getWcs().hashkey(),
                     img.getPsf().hashkey(),
                     self.shapeDev.hashkey(),
                     self.shapeExp.hashkey(),
                     self.fracDev.hashkey()))

    def getParamDerivatives(self, img, modelMask=None):
        e = ExpGalaxy(self.pos, self.brightness, self.shapeExp)
        d = DevGalaxy(self.pos, self.brightness, self.shapeDev)
        e.dname = 'fcomp.exp'
        d.dname = 'fcomp.dev'

        if self.isParamFrozen('pos'):
            e.freezeParam('pos')
            d.freezeParam('pos')
        if self.isParamFrozen('shapeExp'):
            e.freezeParam('shape')
        if self.isParamFrozen('shapeDev'):
            d.freezeParam('shape')

        if hasattr(self, 'halfsize'):
            e.halfsize = self.halfsize
            d.halfsize = self.halfsize

        dexp = e.getParamDerivatives(img, modelMask=modelMask)
        ddev = d.getParamDerivatives(img, modelMask=modelMask)

        # print('FixedCompositeGalaxy.getParamDerivatives.')
        # print('tim shape', img.shape)
        # print('exp deriv extents:')
        # for deriv in dexp + ddev:
        #     print('  ', deriv.name, deriv.getExtent())

        # fracDev scaling
        f = self.fracDev.clipped()
        for deriv in dexp:
            if deriv is not None:
                deriv *= (1. - f)
        for deriv in ddev:
            if deriv is not None:
                deriv *= f

        derivs = []
        i0 = 0
        if not self.isParamFrozen('pos'):
            # "pos" is shared between the models, so add the derivs.
            npos = self.pos.numberOfParams()
            for i in range(npos):
                ii = i0 + i
                dsum = add_patches(dexp[ii], ddev[ii])
                if dsum is not None:
                    dsum.setName('d(fcomp)/d(pos%i)' % i)
                derivs.append(dsum)
            i0 += npos

        if not self.isParamFrozen('brightness'):
            # shared between the models, so add the derivs.
            nb = self.brightness.numberOfParams()
            for i in range(nb):
                ii = i0 + i
                dsum = add_patches(dexp[ii], ddev[ii])
                if dsum is not None:
                    dsum.setName('d(fcomp)/d(bright%i)' % i)
                derivs.append(dsum)
            i0 += nb

        if not self.isParamFrozen('fracDev'):
            counts = img.getPhotoCal().brightnessToCounts(self.brightness)
            if counts == 0.:
                derivs.append(None)
            else:
                # FIXME -- should be possible to avoid recomputing these...
                ue = e.getUnitFluxModelPatch(img, modelMask=modelMask)
                ud = d.getUnitFluxModelPatch(img, modelMask=modelMask)

                df = self.fracDev.derivative()

                if ue is not None:
                    ue *= -df
                if ud is not None:
                    ud *= +df
                df = add_patches(ud, ue)
                if df is None:
                    derivs.append(None)
                else:
                    df *= counts
                    df.setName('d(fcomp)/d(fracDev)')
                    derivs.append(df)

        if not self.isParamFrozen('shapeExp'):
            derivs.extend(dexp[i0:])
        if not self.isParamFrozen('shapeDev'):
            derivs.extend(ddev[i0:])
        return derivs


class CompositeGalaxy(MultiParams, BasicSource):
    '''
    A galaxy with Exponential and deVaucouleurs components.

    The two components share a position (ie the centers are the same),
    but have different brightnesses and shapes.
    '''

    def __init__(self, pos, brightnessExp, shapeExp, brightnessDev, shapeDev):
        MultiParams.__init__(self, pos, brightnessExp, shapeExp,
                             brightnessDev, shapeDev)
        self.name = self.getName()

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightnessExp=1, shapeExp=2,
                    brightnessDev=3, shapeDev=4)

    def getName(self):
        return 'CompositeGalaxy'

    def __str__(self):
        return (self.name + ' at ' + str(self.pos)
                + ' with Exp ' + str(self.brightnessExp) + ' '
                + str(self.shapeExp)
                + ' and deV ' + str(self.brightnessDev) + ' '
                + str(self.shapeDev))

    def __repr__(self):
        return (self.name + '(pos=' + repr(self.pos) +
                ', brightnessExp=' + repr(self.brightnessExp) +
                ', shapeExp=' + repr(self.shapeExp) +
                ', brightnessDev=' + repr(self.brightnessDev) +
                ', shapeDev=' + repr(self.shapeDev))

    def getBrightness(self):
        ''' This makes some assumptions about the
        ``Brightness`` / ``PhotoCal`` and should be treated as
        approximate.'''
        return self.brightnessExp + self.brightnessDev

    def getBrightnesses(self):
        return [self.brightnessExp, self.brightnessDev]

    def _getModelPatches(self, img, minsb=0., modelMask=None):
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        d = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        if minsb == 0. or minsb is None:
            kw = {}
        else:
            kw = dict(minsb=minsb / 2.)
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        pe = e.getModelPatch(img, modelMask=modelMask, **kw)
        pd = d.getModelPatch(img, modelMask=modelMask, **kw)
        return (pe, pd)

    def getModelPatch(self, img, minsb=0., modelMask=None):
        pe, pd = self._getModelPatches(img, minsb=minsb, modelMask=modelMask)
        return add_patches(pe, pd)

    def getUnitFluxModelPatches(self, img, minval=0., modelMask=None):
        # Needed for forced photometry
        if minval > 0:
            # allow each component half the error
            minval = minval * 0.5
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        d = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        ''' 
        out = add_patches(e.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask),
                d.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask))
        return out


        '''
        return (e.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask) +
                d.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask))
        
    # MAGIC: ORDERING OF EXP AND DEV PARAMETERS
    # MAGIC: ASSUMES EXP AND DEV SHAPES SAME LENGTH
    # CompositeGalaxy.
    def getParamDerivatives(self, img, modelMask=None):
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        d = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        e.dname = 'comp.exp'
        d.dname = 'comp.dev'
        if self.isParamFrozen('pos'):
            e.freezeParam('pos')
            d.freezeParam('pos')
        if self.isParamFrozen('brightnessExp'):
            e.freezeParam('brightness')
        if self.isParamFrozen('shapeExp'):
            e.freezeParam('shape')
        if self.isParamFrozen('brightnessDev'):
            d.freezeParam('brightness')
        if self.isParamFrozen('shapeDev'):
            d.freezeParam('shape')

        de = e.getParamDerivatives(img, modelMask=modelMask)
        dd = d.getParamDerivatives(img, modelMask=modelMask)

        if self.isParamFrozen('pos'):
            derivs = de + dd
        else:
            derivs = []
            # "pos" is shared between the models, so add the derivs.
            npos = len(self.pos.getStepSizes())
            for i in range(npos):
                dp = add_patches(de[i], dd[i])
                if dp is not None:
                    dp.setName('d(comp)/d(pos%i)' % i)
                derivs.append(dp)
            derivs.extend(de[npos:])
            derivs.extend(dd[npos:])

        return derivs



class PSFandExpGalaxy(MultiParams, BasicSource):
    '''
    A galaxy with Exponential and deVaucouleurs components.

    The two components share a position (ie the centers are the same),
    but have different brightnesses and shapes.
    '''

    def __init__(self, pos, brightnessExp, shapeExp, brightnessPoint):
        MultiParams.__init__(self, pos, brightnessExp, shapeExp,
                             brightnessPoint)
        self.name = self.getName()

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightnessExp=1, shapeExp=2,
                    brightnessPoint=3)

    def getName(self):
        return 'PSFandExpGalaxy'

    def __str__(self):
        return (self.name + ' at ' + str(self.pos)
                + ' with Exp ' + str(self.brightnessExp) + ' '
                + str(self.shapeExp)
                + ' and Point ' + str(self.brightnessPoint)) 

    def __repr__(self):
        return (self.name + '(pos=' + repr(self.pos) +
                ', brightnessExp=' + repr(self.brightnessExp) +
                ', shapeExp=' + repr(self.shapeExp) +
                ', brightnessPoint=' + repr(self.brightnessPoint))
     
    def getBrightness(self):
        #This makes some assumptions about the
        #``Brightness`` / ``PhotoCal`` and should be treated as
        #approximate.
        return self.brightnessExp + self.brightnessPoint

    def setBrightness(self, brightness):
        self.brightness = self.brightnessExp + self.brightnessPoint
    
    def getBrightnessExp(self):
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp) 
        return e.getBrightness()
  
    def getBrightnessPoint(self):
        d = PointSource(self.pos, self.brightnessPoint)
        return d.getBrightness()

    def getBrightnesses(self):
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        d = PointSource(self.pos, self.brightnessPoint)
        #print(e.getBrightness(),d.getBrightness(),'LOOK HERE BRIGHTNESS')
        return [e.getBrightness(), d.getBrightness()]
        #return [e.brightness, d.brightness]
    
    def _getModelPatches(self, img, minsb=0., modelMask=None):
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        d = PointSource(self.pos, self.brightnessPoint)
        
        if minsb == 0. or minsb is None:
            kw = {}
        else:
            kw = dict(minsb=minsb / 2.)
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
            #print(self.halfsize,'HALFSIZE')
        pe = e.getModelPatch(img, modelMask=modelMask, **kw)
        pd = d.getModelPatch(img, modelMask=modelMask, **kw)
        return (pe, pd)

    def getModelPatch(self, img, minsb=0., modelMask=None):
        pe, pd = self._getModelPatches(img, minsb=minsb, modelMask=modelMask)
        return add_patches(pe, pd)
    
    def _getUnitFluxPatchSize(self, img, px=0., py=0., minval=0.):
        if hasattr(self, 'halfsize'):
            return self.halfsize
        pixscale = img.wcs.pixscale_at(px, py) 
        r = 1.
        s = self.shapeExp
        rexp = max(r, ExpGalaxy.nre * s.re)
        r = max(r, rexp)
        halfsize = r / pixscale
        halfsize += img.psf.getRadius()
        return halfsize

    def getUnitPointFluxModelPatches(self, img, minval=0., modelMask=None):
        # Needed for forced photometry
        if minval > 0:
            # allow each component half the error
            minval = minval * 0.5
        
        d = PointSource(self.pos, self.brightnessPoint)
        return (d.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask))

    def getUnitFluxModelPatches(self, img, minval=0., modelMask=None):
        # Needed for forced photometry
        if minval > 0:
            # allow each component half the error
            minval = minval * 0.5
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        d = PointSource(self.pos, self.brightnessPoint)
        
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        
        patch1 = e.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask)[0]
        patch2 = d.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask)[0] 
        patchout = patch1.patch_sum(patch2)#performArithmetic(patch2,'__sum__')
       
        return ([patchout])
    '''
    def getUnitFluxModelPatches(self, img, minval=0., modelMask=None):
        # Needed for forced photometry
        if minval > 0:
            # allow each component half the error
            minval = minval * 0.5
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        d = PointSource(self.posPoint, self.brightnessPoint)
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        
        return (e.getUnitFluxModelPatches(img, minval=minval,
                                      modelMask=modelMask) +
            d.getUnitFluxModelPatches(img, minval=minval,
                                      modelMask=modelMask))
    '''
    # MAGIC: ORDERING OF EXP AND DEV PARAMETERS
    # MAGIC: ASSUMES EXP AND DEV SHAPES SAME LENGTH
    # CompositeGalaxy.
    def getParamDerivatives(self, img, modelMask=None):
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        d = PointSource(self.pos, self.brightnessPoint)
        #print(self.pos,self.posPoint,'Pos and PosPoint')
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        e.dname = 'comp.exp'
        d.dname = 'comp.point'
        if self.isParamFrozen('pos'):
            e.freezeParam('pos')
            d.freezeParam('pos')
        if self.isParamFrozen('brightnessExp'):
            e.freezeParam('brightness')
        if self.isParamFrozen('shapeExp'):
            e.freezeParam('shape')
        if self.isParamFrozen('brightnessPoint'):
            d.freezeParam('brightness') 

        de = e.getParamDerivatives(img, modelMask=modelMask)
        dd = d.getParamDerivatives(img, modelMask=modelMask)

        if self.isParamFrozen('pos'):
            derivs = de + dd
        else:
            derivs = []
            # "pos" is shared between the models, so add the derivs.
            npos = len(self.pos.getStepSizes()) 
            for i in range(npos):
                dp = add_patches(de[i], dd[i])
                if dp is not None:
                    dp.setName('d(comp)/d(pos%i)' % i)
                derivs.append(dp)
            derivs.extend(de[npos:])
            derivs.extend(dd[npos:])

        return derivs





class PSFandDevGalaxy(MultiParams, BasicSource):
    '''
    A galaxy with Exponential and deVaucouleurs components.

    The two components share a position (ie the centers are the same),
    but have different brightnesses and shapes.
    '''

    def __init__(self, pos, brightnessDev, shapeDev, brightnessPoint):
        MultiParams.__init__(self, pos, brightnessDev, shapeDev,
                             brightnessPoint)
        self.name = self.getName()

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightnessDev=1, shapeDev=2,
                    brightnessPoint=3)

    def getName(self):
        return 'PSFandDevGalaxy'

    def __str__(self):
        return (self.name + ' at ' + str(self.pos)
                + ' with Dev ' + str(self.brightnessDev) + ' '
                + str(self.shapeDev)
                + ' and Point ' + str(self.brightnessPoint)) 

    def __repr__(self):
        return (self.name + '(pos=' + repr(self.pos) +
                ', brightnessDev=' + repr(self.brightnessDev) +
                ', shapeDev=' + repr(self.shapeDev) +
                ', brightnessPoint=' + repr(self.brightnessPoint))
     
    def getBrightness(self):
        #This makes some assumptions about the
        #``Brightness`` / ``PhotoCal`` and should be treated as
        #approximate.
        return self.brightnessDev + self.brightnessPoint

    def setBrightness(self, brightness):
        self.brightness = self.brightnessDev + self.brightnessPoint
    
    def getBrightnessDev(self):
        e = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev) 
        return e.getBrightness()
  
    def getBrightnessPoint(self):
        d = PointSource(self.pos, self.brightnessPoint)
        return d.getBrightness()

    def getBrightnesses(self):
        e = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        d = PointSource(self.pos, self.brightnessPoint)
        #print(e.getBrightness(),d.getBrightness(),'LOOK HERE BRIGHTNESS')
        return [e.getBrightness(), d.getBrightness()]
        #return [e.brightness, d.brightness]
    
    def _getModelPatches(self, img, minsb=0., modelMask=None):
        e = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        d = PointSource(self.pos, self.brightnessPoint)
        
        if minsb == 0. or minsb is None:
            kw = {}
        else:
            kw = dict(minsb=minsb / 2.)
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
            #print(self.halfsize,'HALFSIZE')
        pe = e.getModelPatch(img, modelMask=modelMask, **kw)
        pd = d.getModelPatch(img, modelMask=modelMask, **kw)
        return (pe, pd)

    def getModelPatch(self, img, minsb=0., modelMask=None):
        pe, pd = self._getModelPatches(img, minsb=minsb, modelMask=modelMask)
        return add_patches(pe, pd)
    
    def _getUnitFluxPatchSize(self, img, px=0., py=0., minval=0.):
        if hasattr(self, 'halfsize'):
            return self.halfsize
        pixscale = img.wcs.pixscale_at(px, py) 
        r = 1.
        s = self.shapeDev
        rdev = max(r, DevGalaxy.nre * s.re)
        r = max(r, rdev)
        halfsize = r / pixscale
        halfsize += img.psf.getRadius()
        return halfsize

    def getUnitPointFluxModelPatches(self, img, minval=0., modelMask=None):
        # Needed for forced photometry
        if minval > 0:
            # allow each component half the error
            minval = minval * 0.5
        
        d = PointSource(self.pos, self.brightnessPoint)
        return (d.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask))

    def getUnitFluxModelPatches(self, img, minval=0., modelMask=None):
        # Needed for forced photometry
        if minval > 0:
            # allow each component half the error
            minval = minval * 0.5
        e = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        d = PointSource(self.pos, self.brightnessPoint)
        
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        
        patch1 = e.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask)[0]
        patch2 = d.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask)[0] 
        patchout = patch1.patch_sum(patch2)#performArithmetic(patch2,'__sum__')
       
        return ([patchout])
    '''
    def getUnitFluxModelPatches(self, img, minval=0., modelMask=None):
        # Needed for forced photometry
        if minval > 0:
            # allow each component half the error
            minval = minval * 0.5
        e = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        d = PointSource(self.posPoint, self.brightnessPoint)
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        
        return (e.getUnitFluxModelPatches(img, minval=minval,
                                      modelMask=modelMask) +
            d.getUnitFluxModelPatches(img, minval=minval,
                                      modelMask=modelMask))
    '''
    # MAGIC: ORDERING OF EXP AND DEV PARAMETERS
    # MAGIC: ASSUMES EXP AND DEV SHAPES SAME LENGTH
    # CompositeGalaxy.
    def getParamDerivatives(self, img, modelMask=None):
        e = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        d = PointSource(self.pos, self.brightnessPoint)
        #print(self.pos,self.posPoint,'Pos and PosPoint')
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        e.dname = 'comp.dev'
        d.dname = 'comp.point'
        if self.isParamFrozen('pos'):
            e.freezeParam('pos')
            d.freezeParam('pos')
        if self.isParamFrozen('brightnessDev'):
            e.freezeParam('brightness')
        if self.isParamFrozen('shapeDev'):
            e.freezeParam('shape')
        if self.isParamFrozen('brightnessPoint'):
            d.freezeParam('brightness') 

        de = e.getParamDerivatives(img, modelMask=modelMask)
        dd = d.getParamDerivatives(img, modelMask=modelMask)

        if self.isParamFrozen('pos'):
            derivs = de + dd
        else:
            derivs = []
            # "pos" is shared between the models, so add the derivs.
            npos = len(self.pos.getStepSizes()) 
            for i in range(npos):
                dp = add_patches(de[i], dd[i])
                if dp is not None:
                    dp.setName('d(comp)/d(pos%i)' % i)
                derivs.append(dp)
            derivs.extend(de[npos:])
            derivs.extend(dd[npos:])

        return derivs

class PSFandCompGalaxy(MultiParams, BasicSource):
    '''
    A galaxy with Exponential and deVaucouleurs components.

    The two components share a position (ie the centers are the same),
    but have different brightnesses and shapes.
    '''

    def __init__(self, pos, brightnessExp, shapeExp, brightnessDev, shapeDev, brightnessPoint):
        MultiParams.__init__(self, pos, brightnessExp, shapeExp, brightnessDev, shapeDev,
                             posPoint,brightnessPoint)
        self.name = self.getName()

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightnessExp=1, shapeExp=2, brightnessDev=3, shapeDev=4,
                    brightnessPoint=5)

    def getName(self):
        return 'PSFandCompGalaxy_diffcentres'

    def __str__(self):
        return (self.name + ' at ' + str(self.pos)
                + ' with Exp ' + str(self.brightnessExp) + ' '
                + str(self.shapeExp) 
                + ' with Dev ' + str(self.brightnessDev) + ' '
                + str(self.shapeDev) 
                + ' and Point ' + str(self.brightnessPoint)) 

    def __repr__(self):
        return (self.name + '(pos=' + repr(self.pos) +
                ', brightnessExp=' + repr(self.brightnessExp) +
                ', shapeExp=' + repr(self.shapeExp) +
                ', brightnessDev=' + repr(self.brightnessDev) +
                ', shapeDev=' + repr(self.shapeDev) +
                ', brightnessPoint=' + repr(self.brightnessPoint))
    '''    
    def getBrightness(self):
        #This makes some assumptions about the
        #``Brightness`` / ``PhotoCal`` and should be treated as
        #approximate.
        return self.brightnessExp + self.brightnessPoint
    '''
    def getBrightnesses(self):
      
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        f = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        d = PointSource(self.pos, self.brightnessPoint)
        #print(e,d)
        return [e.getBrightness(), f.getBrightness(), d.getBrightness()]
     
    def getBrightnessPoint(self):
        d = PointSource(self.pos, self.brightnessPoint)
        return d.getBrightness()
    def getBrightnessExp(self):
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp) 
        return e.getBrightness()
    def getBrightnessDev(self):
        f = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev) 
        return f.getBrightness()
  
    def setBrightness(self, brightness):
        self.brightness = self.brightnessExp + self.brightnessDev + self.brightnessPoint
    def getBrightness(self):
        return self.brightnessExp + +selfbrightnessDev, self.brightnessPoint 
    def getPositionPoint(self):
        return self.posPoint
    def _getModelPatches(self, img, minsb=0., modelMask=None):
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        f = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        d = PointSource(self.pos, self.brightnessPoint)
        
        if minsb == 0. or minsb is None:
            kw = {}
        else:
            kw = dict(minsb=minsb / 2.)
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = f.halfsize = self.halfsize
            #print(self.halfsize,'HALFSIZE')
        pe = e.getModelPatch(img, modelMask=modelMask, **kw)
        pd = d.getModelPatch(img, modelMask=modelMask, **kw)
        pf = f.getModelPatch(img, modelMask=modelMask, **kw)
        return (pe, pf, pd)

    def getModelPatch(self, img, minsb=0., modelMask=None):
        pe, pf, pd = self._getModelPatches(img, minsb=minsb, modelMask=modelMask)
        return add_patches(pe, pf, pd)
    
    def getUnitPointFluxModelPatches(self, img, minval=0., modelMask=None):
        # Needed for forced photometry
        if minval > 0:
            # allow each component half the error
             minval = minval * 0.5
        
        d = PointSource(self.posPoint, self.brightnessPoint)
        return (d.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask))


    def getUnitFluxModelPatches(self, img, minval=0., modelMask=None):
        # Needed for forced photometry
        if minval > 0:
            # allow each component half the error
            minval = minval * 0.33
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        f = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        d = PointSource(self.pos, self.brightnessPoint)
        
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        
        patch1 = e.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask)[0]
        patch2 = f.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask)[0] 
        patch3 = d.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask)[0] 

        patchtemp = patch1.patch_sum(patch2)#performArithmetic(patch2,'__sum__')
        patchout = patchtemp.patch_sum(patch3) 
        return ([patchout])
        '''
        return (e.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask) +
                d.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask))
        '''   
    # MAGIC: ORDERING OF EXP AND DEV PARAMETERS
    # MAGIC: ASSUMES EXP AND DEV SHAPES SAME LENGTH
    # CompositeGalaxy.
    def getParamDerivatives(self, img, modelMask=None):
        #print('in derivs function!!')
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        d = PointSource(self.pos, self.brightnessPoint)
        f = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)

        #print(self.pos,self.posPoint,'Pos and PosPoint')
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        e.dname = 'comp.exp'
        f.dname = 'comp.dev'
        d.dname = 'comp.point'
        
        derivs = []
        npos = len(self.pos.getStepSizes())
        
        if self.isParamFrozen('pos'):
            e.freezeParam('pos')
            f.freezeParam('pos')
            d.freezeParam('pos')
        if self.isParamFrozen('brightnessExp'):   
            e.freezeParam('brightness')
        if self.isParamFrozen('brightnessDev'):   
            f.freezeParam('brightness')
        if self.isParamFrozen('shapeExp'):
            e.freezeParam('shape')
        if self.isParamFrozen('shapeDev'):
            f.freezeParam('shape')
        if self.isParamFrozen('brightnessPoint'): 
            d.freezeParam('brightness') 
        
        de = e.getParamDerivatives(img, modelMask=modelMask)
        df = f.getParamDerivatives(img, modelMask=modelMask)
        dd = d.getParamDerivatives(img, modelMask=modelMask)
        
        if self.isParamFrozen('pos'):
            derivs = de + df + dd 
        else:
            # "pos" is shared between the models, so add the derivs.    
            #print(self.pos.getStepSizes(),self.posPoint.getStepSizes()) 
            for i in range(npos):
                dtemp = add_patches(de[i], df[i])
                dp = add_patches(dtemp, dd[i]) #Does dtemp need a [i]????
                if dp is not None:
                    dp.setName('d(comp)/d(pos%i)' % i)
                derivs.append(dp)
            derivs.extend(de[npos:])
            derivs.extend(df[npos:])
            derivs.extend(dd[npos:])
        return derivs



class PSFandDevGalaxy_diffcentres(MultiParams, BasicSource):
    '''
    A galaxy with Exponential and deVaucouleurs components.

    The two components share a position (ie the centers are the same),
    but have different brightnesses and shapes.
    '''

    def __init__(self, pos, brightnessDev, shapeDev, posPoint,brightnessPoint):
        MultiParams.__init__(self, pos, brightnessDev, shapeDev,
                             posPoint,brightnessPoint)
        self.name = self.getName()

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightnessDev=1, shapeDev=2,
                    posPoint=3,brightnessPoint=4)

    def getName(self):
        return 'PSFandDevGalaxy_diffcentres'

    def __str__(self):
        return (self.name + ' at ' + str(self.pos)
                + ' with Dev ' + str(self.brightnessDev) + ' '
                + str(self.shapeDev) + ' at ' + str(self.posPoint)
                + ' and Point ' + str(self.brightnessPoint)) 

    def __repr__(self):
        return (self.name + '(pos=' + repr(self.pos) +
                ', brightnessDev=' + repr(self.brightnessDev) +
                ', shapeDev=' + repr(self.shapeDev) +
                ', brightnessPoint=' + repr(self.brightnessPoint) + ' at ' + str(self.posPoint))
     
    def getBrightness(self):
        #This makes some assumptions about the
        #``Brightness`` / ``PhotoCal`` and should be treated as
        #approximate.
        return self.brightnessDev + self.brightnessPoint

    def setBrightness(self, brightness):
        self.brightness = self.brightnessDev + self.brightnessPoint
    
    def getBrightnessDev(self):
        e = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev) 
        return e.getBrightness()
  
    def getBrightnessPoint(self):
        d = PointSource(self.posPoint, self.brightnessPoint)
        return d.getBrightness()

    def getBrightnesses(self):
        e = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        d = PointSource(self.posPoint, self.brightnessPoint)
        #print(e.getBrightness(),d.getBrightness(),'LOOK HERE BRIGHTNESS')
        return [e.getBrightness(), d.getBrightness()]
        #return [e.brightness, d.brightness]
    
    def _getModelPatches(self, img, minsb=0., modelMask=None):
        e = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        d = PointSource(self.posPoint, self.brightnessPoint)
        
        if minsb == 0. or minsb is None:
            kw = {}
        else:
            kw = dict(minsb=minsb / 2.)
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
            #print(self.halfsize,'HALFSIZE')
        pe = e.getModelPatch(img, modelMask=modelMask, **kw)
        pd = d.getModelPatch(img, modelMask=modelMask, **kw)
        return (pe, pd)

    def getModelPatch(self, img, minsb=0., modelMask=None):
        pe, pd = self._getModelPatches(img, minsb=minsb, modelMask=modelMask)
        return add_patches(pe, pd)
    
    def _getUnitFluxPatchSize(self, img, px=0., py=0., minval=0.):
        if hasattr(self, 'halfsize'):
            return self.halfsize
        pixscale = img.wcs.pixscale_at(px, py) 
        r = 1.
        s = self.shapeDev
        rdev = max(r, DevGalaxy.nre * s.re)
        r = max(r, rdev)
        halfsize = r / pixscale
        halfsize += img.psf.getRadius()
        return halfsize

    def getUnitPointFluxModelPatches(self, img, minval=0., modelMask=None):
        # Needed for forced photometry
        if minval > 0:
            # allow each component half the error
            minval = minval * 0.5
        
        d = PointSource(self.posPoint, self.brightnessPoint)
        return (d.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask))

    def getUnitFluxModelPatches(self, img, minval=0., modelMask=None):
        # Needed for forced photometry
        if minval > 0:
            # allow each component half the error
            minval = minval * 0.5
        e = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        d = PointSource(self.posPoint, self.brightnessPoint)
        
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        
        patch1 = e.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask)[0]
        patch2 = d.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask)[0] 
        patchout = patch1.patch_sum(patch2)#performArithmetic(patch2,'__sum__')
       
        return ([patchout])
    '''
    def getUnitFluxModelPatches(self, img, minval=0., modelMask=None):
        # Needed for forced photometry
        if minval > 0:
            # allow each component half the error
            minval = minval * 0.5
        e = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        d = PointSource(self.posPoint, self.brightnessPoint)
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        
        return (e.getUnitFluxModelPatches(img, minval=minval,
                                      modelMask=modelMask) +
            d.getUnitFluxModelPatches(img, minval=minval,
                                      modelMask=modelMask))
    '''
    # MAGIC: ORDERING OF EXP AND DEV PARAMETERS
    # MAGIC: ASSUMES EXP AND DEV SHAPES SAME LENGTH
    # CompositeGalaxy.
    def getParamDerivatives(self, img, modelMask=None):
        e = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        d = PointSource(self.posPoint, self.brightnessPoint)
        #print(self.pos,self.posPoint,'Pos and PosPoint')
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        e.dname = 'comp.dev'
        d.dname = 'comp.point'
        if self.isParamFrozen('pos'):
            e.freezeParam('pos')
        if self.isParamFrozen('posPoint'):
            d.freezeParam('pos')
        if self.isParamFrozen('brightnessDev'):
            e.freezeParam('brightness')
        if self.isParamFrozen('shapeDev'):
            e.freezeParam('shape')
        if self.isParamFrozen('brightnessPoint'):
            d.freezeParam('brightness') 

        de = e.getParamDerivatives(img, modelMask=modelMask)
        dd = d.getParamDerivatives(img, modelMask=modelMask)

        #if self.isParamFrozen('pos'):
        #    derivs = de + dd
        #else:
        derivs = []
        # "pos" is shared between the models, so add the derivs.
        npos = len(self.pos.getStepSizes()) 
        #for i in range(npos):
        #    dp = add_patches(de[i], dd[i])
        #    if dp is not None:
        #        dp.setName('d(comp)/d(pos%i)' % i)
        #    derivs.append(dp)
        derivs.extend(de)#[npos:])
        derivs.extend(dd)#[npos:])

        return derivs


class PSFandExpGalaxy_diffcentres(MultiParams, BasicSource):
    '''
    A galaxy with Exponential and deVaucouleurs components.

    The two components share a position (ie the centers are the same),
    but have different brightnesses and shapes.
    '''

    def __init__(self, pos, brightnessExp, shapeExp, posPoint,brightnessPoint):
        MultiParams.__init__(self, pos, brightnessExp, shapeExp,
                             posPoint,brightnessPoint)
        self.name = self.getName()

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightnessExp=1, shapeExp=2,
                    posPoint=3,brightnessPoint=4)

    def getName(self):
        return 'PSFandExpGalaxy_diffcentres'

    def __str__(self):
        return (self.name + ' at ' + str(self.pos)
                + ' with Exp ' + str(self.brightnessExp) + ' '
                + str(self.shapeExp) + ' at ' + str(self.posPoint)
                + ' and Point ' + str(self.brightnessPoint)) 

    def __repr__(self):
        return (self.name + '(pos=' + repr(self.pos) +
                ', brightnessExp=' + repr(self.brightnessExp) +
                ', shapeExp=' + repr(self.shapeExp) +
                ', brightnessPoint=' + repr(self.brightnessPoint) + ' at ' + str(self.posPoint))
    '''    
    def getBrightness(self):
        #This makes some assumptions about the
        #``Brightness`` / ``PhotoCal`` and should be treated as
        #approximate.
        return self.brightnessExp + self.brightnessPoint
    '''
    def getBrightnesses(self):
      
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        d = PointSource(self.posPoint, self.brightnessPoint)
        #print(e,d)
        return [e.getBrightness(), d.getBrightness()]
     
    def getBrightnessPoint(self):
        d = PointSource(self.posPoint, self.brightnessPoint)
        return d.getBrightness()
    def getBrightnessExp(self):
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp) 
        return e.getBrightness()
  
    def setBrightness(self, brightness):
        self.brightness = self.brightnessExp + self.brightnessPoint
    def getBrightness(self):
        return self.brightnessExp+self.brightnessPoint 
    def getPositionPoint(self):
        return self.posPoint
    def _getModelPatches(self, img, minsb=0., modelMask=None):
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        d = PointSource(self.posPoint, self.brightnessPoint)
        
        if minsb == 0. or minsb is None:
            kw = {}
        else:
            kw = dict(minsb=minsb / 2.)
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
            #print(self.halfsize,'HALFSIZE')
        pe = e.getModelPatch(img, modelMask=modelMask, **kw)
        pd = d.getModelPatch(img, modelMask=modelMask, **kw)
        return (pe, pd)

    def getModelPatch(self, img, minsb=0., modelMask=None):
        pe, pd = self._getModelPatches(img, minsb=minsb, modelMask=modelMask)
        return add_patches(pe, pd)
    
    def getUnitPointFluxModelPatches(self, img, minval=0., modelMask=None):
        # Needed for forced photometry
        if minval > 0:
            # allow each component half the error
             minval = minval * 0.5
        
        d = PointSource(self.posPoint, self.brightnessPoint)
        return (d.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask))


    def getUnitFluxModelPatches(self, img, minval=0., modelMask=None):
        # Needed for forced photometry
        if minval > 0:
            # allow each component half the error
            minval = minval * 0.5
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        d = PointSource(self.posPoint, self.brightnessPoint)
        print(self.brightnessExp, self.brightnessPoint)
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        
        patch1 = e.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask)[0]
        patch2 = d.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask)[0] 
        patchout = patch1.patch_sum(patch2)#performArithmetic(patch2,'__sum__')
        print(patch1==patchout)
        return ([patchout])
        '''
        return (e.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask) +
                d.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask))
        '''   
    # MAGIC: ORDERING OF EXP AND DEV PARAMETERS
    # MAGIC: ASSUMES EXP AND DEV SHAPES SAME LENGTH
    # CompositeGalaxy.
    def getParamDerivatives(self, img, modelMask=None):
        #print('in derivs function!!')
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        d = PointSource(self.posPoint, self.brightnessPoint)
        #print(self.pos,self.posPoint,'Pos and PosPoint')
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        e.dname = 'comp.exp'
        d.dname = 'comp.point'
        
        derivs = []
        npos = len(self.pos.getStepSizes())
        
        if self.isParamFrozen('pos'):
            e.freezeParam('pos')
        if self.isParamFrozen('posPoint'):
            d.freezeParam('pos')
        if self.isParamFrozen('brightnessExp'):
            #print('brightnessExp is frozen')
            e.freezeParam('brightnessExp')
        if self.isParamFrozen('shapeExp'):
            e.freezeParam('shape')
        if self.isParamFrozen('brightnessPoint'):
            #print('brightness Point is frozen')
            d.freezeParam('brightness') 
        #print(self.pos,self.posPoint)
        de = e.getParamDerivatives(img, modelMask=modelMask)
        dd = d.getParamDerivatives(img, modelMask=modelMask)
        derivs = de + dd
        #if self.isParamFrozen('pos'):
        #    derivs = de + dd
        #else:
        
        # "pos" is shared between the models, so add the derivs.
       
        #print(self.pos.getStepSizes(),self.posPoint.getStepSizes()) 
        #for i in range(npos):
        #    dp = add_patches(de[i], dd[i])
        #    if dp is not None:
        #        dp.setName('d(comp)/d(pos%i)' % i)
        #    derivs.append(dp)
        #derivs.extend(de)#[npos:])
        #derivs.extend(dd)#[npos:])

        return derivs


class PSFandCompGalaxy_diffcentres(MultiParams, BasicSource):
    '''
    A galaxy with Exponential and deVaucouleurs components.

    The two components share a position (ie the centers are the same),
    but have different brightnesses and shapes.
    '''

    def __init__(self, pos, brightnessExp, shapeExp, brightnessDev, shapeDev, posPoint,brightnessPoint):
        MultiParams.__init__(self, pos, brightnessExp, shapeExp, brightnessDev, shapeDev,
                             posPoint,brightnessPoint)
        self.name = self.getName()

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightnessExp=1, shapeExp=2, brightnessDev=3, shapeDev=4,
                    posPoint=5,brightnessPoint=6)

    def getName(self):
        return 'PSFandCompGalaxy_diffcentres'

    def __str__(self):
        return (self.name + ' at ' + str(self.pos)
                + ' with Exp ' + str(self.brightnessExp) + ' '
                + str(self.shapeExp) 
                + ' with Dev ' + str(self.brightnessDev) + ' '
                + str(self.shapeDev) 
                + ' at ' + str(self.posPoint)
                + ' and Point ' + str(self.brightnessPoint)) 

    def __repr__(self):
        return (self.name + '(pos=' + repr(self.pos) +
                ', brightnessExp=' + repr(self.brightnessExp) +
                ', shapeExp=' + repr(self.shapeExp) +
                ', brightnessDev=' + repr(self.brightnessDev) +
                ', shapeDev=' + repr(self.shapeDev) +
                ', brightnessPoint=' + repr(self.brightnessPoint) + ' at ' + str(self.posPoint))
    '''    
    def getBrightness(self):
        #This makes some assumptions about the
        #``Brightness`` / ``PhotoCal`` and should be treated as
        #approximate.
        return self.brightnessExp + self.brightnessPoint
    '''
    def getBrightnesses(self):
      
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        f = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        d = PointSource(self.posPoint, self.brightnessPoint)
        #print(e,d)
        return [e.getBrightness(), f.getBrightness(), d.getBrightness()]
     
    def getBrightnessPoint(self):
        d = PointSource(self.posPoint, self.brightnessPoint)
        return d.getBrightness()
    def getBrightnessExp(self):
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp) 
        return e.getBrightness()
    def getBrightnessDev(self):
        f = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev) 
        return f.getBrightness()
  
    def setBrightness(self, brightness):
        self.brightness = self.brightnessExp + self.brightnessDev + self.brightnessPoint
    def getBrightness(self):
        return self.brightnessExp + +selfbrightnessDev, self.brightnessPoint 
    def getPositionPoint(self):
        return self.posPoint
    def _getModelPatches(self, img, minsb=0., modelMask=None):
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        f = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        d = PointSource(self.posPoint, self.brightnessPoint)
        
        if minsb == 0. or minsb is None:
            kw = {}
        else:
            kw = dict(minsb=minsb / 2.)
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = f.halfsize = self.halfsize
            #print(self.halfsize,'HALFSIZE')
        pe = e.getModelPatch(img, modelMask=modelMask, **kw)
        pd = d.getModelPatch(img, modelMask=modelMask, **kw)
        pf = f.getModelPatch(img, modelMask=modelMask, **kw)
        return (pe, pf, pd)

    def getModelPatch(self, img, minsb=0., modelMask=None):
        pe, pf, pd = self._getModelPatches(img, minsb=minsb, modelMask=modelMask)
        return add_patches(pe, pf, pd)
    
    def getUnitPointFluxModelPatches(self, img, minval=0., modelMask=None):
        # Needed for forced photometry
        if minval > 0:
            # allow each component half the error
             minval = minval * 0.5
        
        d = PointSource(self.posPoint, self.brightnessPoint)
        return (d.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask))


    def getUnitFluxModelPatches(self, img, minval=0., modelMask=None):
        # Needed for forced photometry
        if minval > 0:
            # allow each component half the error
            minval = minval * 0.5
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        f = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        d = PointSource(self.posPoint, self.brightnessPoint)
        
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        
        patch1 = e.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask)[0]
        patch2 = f.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask)[0] 
        patch3 = d.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask)[0] 

        patchtemp = patch1.patch_sum(patch2)#performArithmetic(patch2,'__sum__')
        patchout = patchtemp.patch_sum(patch3) 
        return ([patchout])
        '''
        return (e.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask) +
                d.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask))
        '''   
    # MAGIC: ORDERING OF EXP AND DEV PARAMETERS
    # MAGIC: ASSUMES EXP AND DEV SHAPES SAME LENGTH
    # CompositeGalaxy.
    def getParamDerivatives(self, img, modelMask=None):
        #print('in derivs function!!')
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        d = PointSource(self.posPoint, self.brightnessPoint)
        f = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)

        #print(self.pos,self.posPoint,'Pos and PosPoint')
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        e.dname = 'comp.exp'
        f.dname = 'comp.dev'
        d.dname = 'comp.point'
        
        derivs = []
        npos = len(self.pos.getStepSizes())
        
        if self.isParamFrozen('pos'):
            e.freezeParam('pos')
            f.freezeParam('pos')
        if self.isParamFrozen('posPoint'):
            d.freezeParam('pos')
        if self.isParamFrozen('brightnessExp'):   
            e.freezeParam('brightness')
        if self.isParamFrozen('brightnessDev'):   
            f.freezeParam('brightness')
        if self.isParamFrozen('shapeExp'):
            e.freezeParam('shape')
        if self.isParamFrozen('shapeDev'):
            f.freezeParam('shape')
        if self.isParamFrozen('brightnessPoint'): 
            d.freezeParam('brightness') 
        
        de = e.getParamDerivatives(img, modelMask=modelMask)
        df = f.getParamDerivatives(img, modelMask=modelMask)
        dd = d.getParamDerivatives(img, modelMask=modelMask)
        
        if self.isParamFrozen('pos'):
            derivs = de + df + dd 
        else:
            # "pos" is shared between the models, so add the derivs.    
            #print(self.pos.getStepSizes(),self.posPoint.getStepSizes()) 
            for i in range(npos):
                dp = add_patches(de[i], df[i])
                if dp is not None:
                    dp.setName('d(comp)/d(pos%i)' % i)
                derivs.append(dp)
            derivs.extend(de)[npos:])
            derivs.extend(df)[npos:])
            derivs.extend(dd)
        return derivs

class TwoPSFandDevGalaxy_diffcentres(MultiParams, BasicSource):
    '''
    A galaxy with Exponential and deVaucouleurs components.

    The two components share a position (ie the centers are the same),
    but have different brightnesses and shapes.
    '''

    def __init__(self, pos, brightnessDev, shapeDev, posPoint1, brightnessPoint1, posPoint2, brightnessPoint2):
        MultiParams.__init__(self, pos, brightnessDev, shapeDev,
                             posPoint1,brightnessPoint1,posPoint2,brightnessPoint2)
        self.name = self.getName()

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightnessDev=1, shapeDev=2,
                    posPoint1=3,brightnessPoint1=4,posPoint1=5,brightnessPoint1=6)

    def getName(self):
        return 'TwoPSFandDevGalaxy_diffcentres'

    def __str__(self):
        return (self.name + ' at ' + str(self.pos)
                + ' with Dev ' + str(self.brightnessDev) + ' '
                + str(self.shapeDev) + ' at ' + str(self.posPoint1)
                + ' and Point ' + str(self.brightnessPoint1)
                + ' and Point ' + str(self.brightnessPoint2) + ' at ' + str(self.posPoint2)) 

    def __repr__(self):
        return (self.name + '(pos=' + repr(self.pos) +
                ', brightnessDev=' + repr(self.brightnessDev) +
                ', shapeDev=' + repr(self.shapeDev) +
                ', brightnessPoint=' + repr(self.brightnessPoint) + ' at ' + str(self.posPoint)
                + ' and Point ' + str(self.brightnessPoint2) + ' at ' + str(self.posPoint2))
     
    def getBrightness(self):
        #This makes some assumptions about the
        #``Brightness`` / ``PhotoCal`` and should be treated as
        #approximate.
        return self.brightnessDev + self.brightnessPoint1 + self.brightnessPoint2

    def setBrightness(self, brightness):
        self.brightness = self.brightnessDev + self.brightnessPoint1 + self.brightnessPoint2
    
    def getBrightnessDev(self):
        e = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev) 
        return e.getBrightness()
  
    def getBrightnessPoint1(self):
        d = PointSource(self.posPoint1, self.brightnessPoint1)
        return d.getBrightness()

    def getBrightnessPoint2(self):
        f = PointSource(self.posPoint2, self.brightnessPoint2)
        return d.getBrightness()


    def getBrightnesses(self):
        e = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        d = PointSource(self.posPoint1, self.brightnessPoint1)
        f = PointSource(self.posPoint2, self.brightnessPoint2)

        #print(e.getBrightness(),d.getBrightness(),'LOOK HERE BRIGHTNESS')
        return [e.getBrightness(), d.getBrightness(), f.getBrightness()]
        #return [e.brightness, d.brightness]
    
    def _getModelPatches(self, img, minsb=0., modelMask=None):
        e = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        d = PointSource(self.posPoint1, self.brightnessPoint1)
        f = PointSource(self.posPoint2, self.brightnessPoint2)
 
        if minsb == 0. or minsb is None:
            kw = {}
        else:
            kw = dict(minsb=minsb / 2.)
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
            #print(self.halfsize,'HALFSIZE')
        pe = e.getModelPatch(img, modelMask=modelMask, **kw)
        pd = d.getModelPatch(img, modelMask=modelMask, **kw)
        pf = f.getModelPatch(img, modelMask=modelMask, **kw)

        return (pe, pd, pf)

    def getModelPatch(self, img, minsb=0., modelMask=None):
        pe, pd, pf = self._getModelPatches(img, minsb=minsb, modelMask=modelMask)
        patch_temp = add_patches(pe, pd)
        return add_patches(patch_temp, pf)
        
    def _getUnitFluxPatchSize(self, img, px=0., py=0., minval=0.):
        if hasattr(self, 'halfsize'):
            return self.halfsize
        pixscale = img.wcs.pixscale_at(px, py) 
        r = 1.
        s = self.shapeDev
        rdev = max(r, DevGalaxy.nre * s.re)
        r = max(r, rdev)
        halfsize = r / pixscale
        halfsize += img.psf.getRadius()
        return halfsize

    def getUnitPointFluxModelPatches(self, img, minval=0., modelMask=None):
        # Needed for forced photometry
        if minval > 0:
            # allow each component half the error
            minval = minval * 0.5
        
        d = PointSource(self.posPoint, self.brightnessPoint)
        return (d.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask))

    def getUnitFluxModelPatches(self, img, minval=0., modelMask=None):
        # Needed for forced photometry
        if minval > 0:
            # allow each component half the error
            minval = minval * 0.5
        e = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        d = PointSource(self.posPoint1, self.brightnessPoint1)
        f = PointSource(self.posPoint2, self.brightnessPoint2)

        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        
        patch1 = e.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask)[0]
        patch2 = d.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask)[0] 
        patch3 = f.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask)[0] 

        
        patchtemp = patch1.patch_sum(patch2)#performArithmetic(patch2,'__sum__')
        patchout = patchtemp.patch_sum(patch3)
        return ([patchout])
    # MAGIC: ORDERING OF EXP AND DEV PARAMETERS
    # MAGIC: ASSUMES EXP AND DEV SHAPES SAME LENGTH
    # CompositeGalaxy.
    def getParamDerivatives(self, img, modelMask=None):
        e = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        d = PointSource(self.posPoint1, self.brightnessPoint1)
        f = PointSource(self.posPoint2, self.brightnessPoint2)

        #print(self.pos,self.posPoint,'Pos and PosPoint')
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = f.halfsize = self.halfsize
        e.dname = 'dev.dev'
        d.dname = 'dev.point1'
        f.dname = 'dev.point2'
        if self.isParamFrozen('pos'):
            e.freezeParam('pos')
        if self.isParamFrozen('posPoint1'):
            d.freezeParam('pos')
        if self.isParamFrozen('posPoint2'):
            f.freezeParam('pos')    
        if self.isParamFrozen('brightnessDev'):
            e.freezeParam('brightness')
        if self.isParamFrozen('shapeDev'):
            e.freezeParam('shape')
        if self.isParamFrozen('brightnessPoint1'):
            d.freezeParam('brightness') 
        if self.isParamFrozen('brightnessPoint2'):
            f.freezeParam('brightness') 


        de = e.getParamDerivatives(img, modelMask=modelMask)
        dd = d.getParamDerivatives(img, modelMask=modelMask)
        df = f.getParamDerivatives(img, modelMask=modelMask)

        #if self.isParamFrozen('pos'):
        #    derivs = de + dd
        #else:
        derivs = []
        # "pos" is shared between the models, so add the derivs.
        npos = len(self.pos.getStepSizes()) 
        #for i in range(npos):
        #    dp = add_patches(de[i], dd[i])
        #    if dp is not None:
        #        dp.setName('d(comp)/d(pos%i)' % i)
        #    derivs.append(dp)
        derivs.extend(de)#[npos:])
        derivs.extend(dd)#[npos:])
        derivs.extend(df)
        return derivs



class TwoPSFandExpGalaxy_diffcentres(MultiParams, BasicSource):
    '''
    A galaxy with Exponential and deVaucouleurs components.

    The two components share a position (ie the centers are the same),
    but have different brightnesses and shapes.
    '''

    def __init__(self, pos, brightnessExp, shapeExp, posPoint1, brightnessPoint1, posPoint2, brightnessPoint2):
        MultiParams.__init__(self, pos, brightnessExp, shapeExp,
                             posPoint1,brightnessPoint1,posPoint2,brightnessPoint2)
        self.name = self.getName()

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightnessExp=1, shapeExp=2,
                    posPoint1=3,brightnessPoint1=4,posPoint1=5,brightnessPoint1=6)

    def getName(self):
        return 'TwoPSFandExpGalaxy_diffcentres'

    def __str__(self):
        return (self.name + ' at ' + str(self.pos)
                + ' with Exp ' + str(self.brightnessExp) + ' '
                + str(self.shapeExp) + ' at ' + str(self.posPoint1)
                + ' and Point ' + str(self.brightnessPoint1)
                + ' and Point ' + str(self.brightnessPoint2) + ' at ' + str(self.posPoint2)) 

    def __repr__(self):
        return (self.name + '(pos=' + repr(self.pos) +
                ', brightnessExp=' + repr(self.brightnessExp) +
                ', shapeExp=' + repr(self.shapeExp) +
                ', brightnessPoint=' + repr(self.brightnessPoint) + ' at ' + str(self.posPoint)
                + ' and Point ' + str(self.brightnessPoint2) + ' at ' + str(self.posPoint2))
     
    def getBrightness(self):
        #This makes some assumptions about the
        #``Brightness`` / ``PhotoCal`` and should be treated as
        #approximate.
        return self.brightnessExp + self.brightnessPoint1 + self.brightnessPoint2

    def setBrightness(self, brightness):
        self.brightness = self.brightnessExp + self.brightnessPoint1 + self.brightnessPoint2
    
    def getBrightnessExp(self):
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp) 
        return e.getBrightness()
  
    def getBrightnessPoint1(self):
        d = PointSource(self.posPoint1, self.brightnessPoint1)
        return d.getBrightness()

    def getBrightnessPoint2(self):
        f = PointSource(self.posPoint2, self.brightnessPoint2)
        return d.getBrightness()


    def getBrightnesses(self):
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        d = PointSource(self.posPoint1, self.brightnessPoint1)
        f = PointSource(self.posPoint2, self.brightnessPoint2)

        #print(e.getBrightness(),d.getBrightness(),'LOOK HERE BRIGHTNESS')
        return [e.getBrightness(), d.getBrightness(), f.getBrightness()]
        #return [e.brightness, d.brightness]
    
    def _getModelPatches(self, img, minsb=0., modelMask=None):
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        d = PointSource(self.posPoint1, self.brightnessPoint1)
        f = PointSource(self.posPoint2, self.brightnessPoint2)
 
        if minsb == 0. or minsb is None:
            kw = {}
        else:
            kw = dict(minsb=minsb / 2.)
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
            #print(self.halfsize,'HALFSIZE')
        pe = e.getModelPatch(img, modelMask=modelMask, **kw)
        pd = d.getModelPatch(img, modelMask=modelMask, **kw)
        pf = f.getModelPatch(img, modelMask=modelMask, **kw)

        return (pe, pd, pf)

    def getModelPatch(self, img, minsb=0., modelMask=None):
        pe, pd, pf = self._getModelPatches(img, minsb=minsb, modelMask=modelMask)
        patch_temp = add_patches(pe, pd)
        return add_patches(patch_temp, pf)
        
    def _getUnitFluxPatchSize(self, img, px=0., py=0., minval=0.):
        if hasattr(self, 'halfsize'):
            return self.halfsize
        pixscale = img.wcs.pixscale_at(px, py) 
        r = 1.
        s = self.shapeExp
        rexp = max(r, ExpGalaxy.nre * s.re)
        r = max(r, rexp)
        halfsize = r / pixscale
        halfsize += img.psf.getRadius()
        return halfsize

    def getUnitPointFluxModelPatches(self, img, minval=0., modelMask=None):
        # Needed for forced photometry
        if minval > 0:
            # allow each component half the error
            minval = minval * 0.5
        
        d = PointSource(self.posPoint, self.brightnessPoint)
        return (d.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask))

    def getUnitFluxModelPatches(self, img, minval=0., modelMask=None):
        # Needed for forced photometry
        if minval > 0:
            # allow each component half the error
            minval = minval * 0.5
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        d = PointSource(self.posPoint1, self.brightnessPoint1)
        f = PointSource(self.posPoint2, self.brightnessPoint2)

        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        
        patch1 = e.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask)[0]
        patch2 = d.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask)[0] 
        patch3 = f.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask)[0] 

        
        patchtemp = patch1.patch_sum(patch2)#performArithmetic(patch2,'__sum__')
        patchout = patchtemp.patch_sum(patch3)
        return ([patchout])
    # MAGIC: ORDERING OF EXP AND DEV PARAMETERS
    # MAGIC: ASSUMES EXP AND DEV SHAPES SAME LENGTH
    # CompositeGalaxy.
    def getParamDerivatives(self, img, modelMask=None):
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        d = PointSource(self.posPoint1, self.brightnessPoint1)
        f = PointSource(self.posPoint2, self.brightnessPoint2)

        #print(self.pos,self.posPoint,'Pos and PosPoint')
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = f.halfsize = self.halfsize
        e.dname = 'exp.exp'
        d.dname = 'exp.point1'
        f.dname = 'exp.point2'
        if self.isParamFrozen('pos'):
            e.freezeParam('pos')
        if self.isParamFrozen('posPoint1'):
            d.freezeParam('pos')
        if self.isParamFrozen('posPoint2'):
            f.freezeParam('pos')    
        if self.isParamFrozen('brightnessExp'):
            e.freezeParam('brightness')
        if self.isParamFrozen('shapeExp'):
            e.freezeParam('shape')
        if self.isParamFrozen('brightnessPoint1'):
            d.freezeParam('brightness') 
        if self.isParamFrozen('brightnessPoint2'):
            f.freezeParam('brightness') 


        de = e.getParamDerivatives(img, modelMask=modelMask)
        dd = d.getParamDerivatives(img, modelMask=modelMask)
        df = f.getParamDerivatives(img, modelMask=modelMask)

        #if self.isParamFrozen('pos'):
        #    derivs = de + dd
        #else:
        derivs = []
        # "pos" is shared between the models, so add the derivs.
        npos = len(self.pos.getStepSizes()) 
        #for i in range(npos):
        #    dp = add_patches(de[i], dd[i])
        #    if dp is not None:
        #        dp.setName('d(comp)/d(pos%i)' % i)
        #    derivs.append(dp)
        derivs.extend(de)#[npos:])
        derivs.extend(dd)#[npos:])
        derivs.extend(df)
        return derivs


