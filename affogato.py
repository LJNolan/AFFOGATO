#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 13:54:29 2025
Original File Created on Mon Feb 20 1:32:29 2023

Author: Liam Nolan

AFFOGATO: gAlactic Faint Feature extractiOn with GALFITM-bAsed Tools in pythOn

Based in large part on Tony Chen's GALFIT wrapper, and suggestions from
Ming-Yang Zhuang.

General Notes for Use:
These wrapper functions generally assume all of the GALFITM work is being done 
in a fairly clean directory, where files are not being saved permanently, 
instead being backed up somewhere else if needed, and other intermediate 
processes are discarded. I've tried to make these functions as general as is
practical, and noted in comments where the code becomes specific to my use 
case.
All filenames should be as relative to the location where this wrapper is
stored unless specified otherwise.

NOTA BENE: On my Mac, the GALFITM executable did not work until placing it in
the src folder of Anaconda3, and only then did it throw an error stating it was
being blocked by the secuity software of my Mac.  I was then able to override
that block in the Privacy & Security settings.  I provide options for where to
send your download, but your mileage may vary.
"""

import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from astropy.io import ascii, fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.table import Table, join
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization import LogStretch, astropy_mpl_style
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.utils.data import get_pkg_data_filename
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.convolution import convolve
from astroquery.mast import Observations
from photutils.segmentation import detect_threshold, detect_sources, \
                                   deblend_sources, SourceCatalog, \
                                   make_2dgaussian_kernel
from photutils.utils import circular_footprint
from photutils.profiles import RadialProfile
from scipy.ndimage import convolve
import urllib.request

plt.style.use(astropy_mpl_style)


class NoWCSError(Exception):
    pass


def cut(filename, position, size, ext=1, outputdir=".", outputname=None):
   """
   Create a cutout image of given ``size`` at given ``position`` and save it
   with correct WCS and several copied keywords.
   Designed for use with HST keywords.

   Parameters
   ----------
   filename : string
      Name of the FITS image from which to cut, as relative to the current
      directory - for example 'data/ie4701_drz.fits'
   
   position : SkyCoord
      Position of target / center of cut.
   
   size : tuple (int)
      Pixel size of desired cutout.
   
   ext : int, optional
      The desired extension from which to cut data (data will always be saved
      to extension 1 in the output file). The default is 1 [SCI].
   
   outputdir : string, optional
      The directory to output the cutout image. In the context of this
      pipeline, this should be the clean working directory described above. The
      default is blank, i.e. the location of this file.
   
   outputname : str, optional
      The desired name of the cutout image. The default is 'image.fits'. 

   Returns
   -------
   None.

   Notes
   -----
   Produces a cutout image ``outputname``.

   """
   
   # Load the image and the WCS
   with fits.open(filename) as hdul:
      wcs = WCS(hdul[ext].header)

      # Make the cutout, including the WCS
      cutout = Cutout2D(hdul[ext].data, position=position, size=size, wcs=wcs)

      # Put the cutout image in the FITS HDU
      hdul_new = fits.PrimaryHDU(cutout.data)

      # Update the FITS header with the cutout WCS
      hdr_new = hdul_new.header
      hdr_new.update(cutout.wcs.to_header())
    
      ## source info
      hdr_new['PROPOSID'] = hdul[0].header['PROPOSID']
      hdr_new['TARGNAME'] = hdul[0].header['TARGNAME']
      hdr_new['RA_TARG'] = hdul[0].header['RA_TARG']
      hdr_new['DEC_TARG'] = hdul[0].header['DEC_TARG']
      ## obs info
      hdr_new['GAIN'] = 1 # Only true if data is in units of electrons
      hdr_new['NCOMBINE'] = 1 # Only true if EXPTIME is over all exposures
      hdr_new['EXPTIME'] = hdul[0].header['EXPTIME']
      # TODO: Check if this switch 0 -> ext works for all files
      hdr_new['photzpt'] = hdul[ext].header['photzpt']
      hdr_new['photflam'] = hdul[ext].header['photflam']
      hdr_new['photplam'] = hdul[ext].header['photplam']
      ## drizzle info
      hdr_new['fscale'] = hdul[0].header['D001SCAL']
   
      # Write the cutout to a new FITS file
      if outputname is None:
         if ext == 1:
            hdul_new.writeto('%s/image.fits' % outputdir, overwrite=True)
         else:
            hdul_new.writeto('%s/image%i.fits' % (outputdir, ext),
                             overwrite=True)
      else:
         hdul_new.writeto('%s/%s' % (outputdir, outputname), overwrite=True)
   return


def run_galfitm(galpath='me', outputdir='.'):
   """
   Runs GALFITM off of the file "``outputdir``/input" in option 3 mode, which
   creates a FITS image where each slice is a model component, called
   "subcomps.fits". This will clear all previous runs, but will preserve the
   fit log.  Use bkp_galfitm() to back up the run.

   Parameters
   ----------
   galpath : string, optional
      Directory path of where the user has GALFITM installed. The default is 
      'me', which directs to the installation on my current computer.
   
   outputdir : string, optional
      Directory path of where to run GALFITM. The default is '.'.

   Returns
   -------
   None.
   
   Notes
   -----
   Produces and overwrites galfitm.fits, subcomps.fits, fit.tab, and appends to
   fit.log.

   """
   if galpath == 'me':
      galpath = '/Users/ljnolan/opt/anaconda3/bin/galfitm'
   os.system("cd %s ; rm -f galfitm.*" % outputdir)
   os.system("cd %s ; " % outputdir + galpath + " input")
   return


def bkp_galfitm(todir, fromdir='.', name='', gal=True, inpt=False,
               log=False, image=None):
   """
   Backs up the outputs of GALFITM, with multiple toggles for different outputs
   to back up. By default, backs up the normal GALFITM outputs.
   
   Parameters
   ----------
   todir : string
      Directory to which files should be backed up. Must be as relative to
      ``fromdir`` if supplied.
   
   fromdir : string, optional
      Directory from which files come - this should be the directory where
      run_galfitm() was run. The default is '.'.
   
   name : string, optional
      String to prepend to GALFITM files - typically something that identifies
      the run/object. The default is '', which results in no prepending.
   
   gal : boolean, optional
      Toggle to back up 'galfitm.galfit.01' and 'galfitm.fits'. The default is True.
   
   inpt : boolean, optional
      Toggle to back up 'input'. The default is False.
   
   log : boolean, optional
      Toggle to back up 'fit.log'. The default is False.
      Note that 'fit.log' logs all fits since it was last cleared.
   
   image : string, optional
      File to be backed up - technically can be anything, but intended to be an
      output image. The default is None.

   Returns
   -------
   None.

   """
   if name != '':
      name += '.'
   if gal:
      os.system("cd %s ; cp galfitm.galfit.01 %s/%sgalfitm.galfit.01" % (fromdir, todir, 
                                                          name))
      os.system("cd %s ; cp galfitm.fits %s/%sgalfitm.fits" % (fromdir, todir, 
                                                              name))
   if inpt:
      os.system("cd %s ; cp input %s/%sinput" % (fromdir, todir, name))
   if log:
      os.system("cd %s ; cp fit.log %s/%sfit.log" % (fromdir, todir, name))
   if image is None:
      pass
   else:
      os.system("cd %s ; cp %s %s/%s%s" % (fromdir, image, todir, name, 
                                            image))
   return


def clearlog(path='.'):
   """
   Deletes the current GALFITM log file "fit.log" in ``path``

   Parameters
   ----------
   path : string, optional
      The directory where to clear the log. The default is '.'.

   Returns
   -------
   None.

   """
   os.system("cd %s ; rm -f fit.log" % path)
   return


def dataPull(file, ext=0):
   """
   Extracts the data from a FITS file and returns it in an array.

   Parameters
   ----------
   file : string
      The path to the FITS file from which data is to be extracted.
      
   ext : int, optional
      FITS extension of data. The default is 0.

   Returns
   -------
   image_data : array-like
      The data from the FITS file and extension.

   """
   image = get_pkg_data_filename(file)
   image_data = fits.getdata(image, ext=ext)
   return image_data


def getBkgrd(image, ext=0, sigma=3.0, npix=50):
   """
   Estimates the background level of the image.

   Parameters
   ----------
   image : string or array-like
      FITS file path from which background should be estimated, or the data
      from that file.
   ext : int, optional
      FITS extension from which background should be estimated. The default is
      0.
   sigma : float, optional
      Sigma level at which to clip data before estimating background. The 
      default is 3.0.
   npix : int, optional
      Number of continuous pixels required for a source which will be masked
      before estimation. The default is 50.

   Returns
   -------
   background : float
      The estimated background level of the image.

   """
   if type(image) == str:
      data = dataPull(image, ext)
   else:
      data = image

   sigma_clip = SigmaClip(sigma=sigma, maxiters=10)
   threshold = detect_threshold(data, nsigma=3.0, sigma_clip=sigma_clip)
   segment_img = detect_sources(data, threshold, npixels=npix)

   footprint = circular_footprint(radius=10)
   mask = segment_img.make_source_mask(footprint=footprint)

   mean, median, std = sigma_clipped_stats(data, sigma=sigma, mask=mask)
   return float(mean)


def make_galfitm_input(img_size, zp, img_scale, sky, comps, outputdir=".",
                       fsf=4):
   """
   Takes in summary information on the current image to be operated upon, and
   generates input files for GALFITM.

   Parameters
   ----------
   img_size : tuple, int
      X, Y dimensions of input image, in pixels.
      
   zp : float
      Magnitude photometric zeropoint of image.
      
   img_scale : tuple, float
      Plate scale (dx, dy) in arcsec per pixel.
      
   sky : float
      Image sky background value in ADU counts.
      
   comps : list, mixed types
      A list describing the initial guesses for the model components s follows:
         * if a component is a PSF, the list element is a list of length 3 of
           floats: x, y, magnitude
         * if a component is something else, the list element is a list of 
           length 8 of str what component type it is, followed by seven floats:
           x, y, mag, length, index, axis ratio, and angle
      This can be generated by input_to_guess() or get_guesses().
      
   outputdir : str, optional
      Where to save the output file. The default is ".", i.e. the working
      directory.
   
   fsf : int, optional
      Fine-sampling factor for the PSF, relative to the data.  The default is
      4, appropriate for the standard PSF used in the quick-look functions.

   Returns
   -------
   None.
   
   Notes
   -----
   Produces two files, 'input' and constraints', by default in the current
   directory, suitable for running GALFITM.'
   
   I make a number of assumptions/simplifications, suitable for the purposes of
   my work, but perhaps would be undesireable for other purposes.  I enumerate
   some here:
      * Convolution is over the entire (cutout) image.
      * The sky background is a good estimate and should not be allowed to vary
        in the fitting (easy to change)
      * The reading in of "comps" is really only designed for PSFs and sersics
        - other components are TBD
      * The function will overwrite an existing file with the same name
      * The constraint file is very hands-off, but prevents values from 
        generally becoming unphysical.  The "length" value should be liberally
        altered - it can generally be pretty tightly constrained from visual
        inspection.  This also means one should keep an eye out for parameters
        hitting their bounds, this is likely a poor fit.
      * I round all the floats in the input file except zeropoint, because in
        my experience not rounding just gives a bunch of useless zeroes. If you
        don't want this, just find-replace '%0.2f' -> '%f'
      * There is a file 'psf.fits' in the output directory for PSF convolution
      * The PSF fine sampling factor (E) is 3 (easy to change)

   """
   os.system("cd %s ; rm -f input" % outputdir)
   os.system("cd %s ; rm -f constraints" % outputdir)
   
   string_init = """
===============================================================================
# IMAGE and GALFITM CONTROL PARAMETERS
A) image.fits      # Input data image (FITS file)
B) galfitm.fits     # Output data image block
C) none            # Sigma image name (made from data if blank or "none")
D) psf.fits        # Input PSF image and (optional) diffusion kernel
E) %i               # PSF fine sampling factor relative to data
F) mask.fits       # Bad pixel mask (FITS image or ASCII coord list)
G) constraints     # File with parameter constraints (ASCII file) 
H) 1 %i 1 %i     # Image region to fit (xmin xmax ymin ymax)
I) %i %i         # Size of the convolution box (x y)
J) %f       # Magnitude photometric zeropoint 
K) %0.2f %0.2f       # Plate scale (dx dy)   [arcsec per pixel]
O) regular         # Display type (regular, curses, both)
P) 0               # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps
W) input,model,residual,component      # Output options
    """

   string_sky = """
 0) sky
 1) %0.2f    0        # sky background       [ADU counts]
 2) 0.00      0        # dsky/dx (sky gradient in x) 
 3) 0.00      0        # dsky/dy (sky gradient in x) 
 Z) 0                  #  Skip this model in output image?  (yes=1, no=0)
    """

   string_psf = """
 0) psf                # object type
 1) %0.2f %0.2f  1 1  # position x, y        [pixel]
 3) %0.2f     1        # total magnitude
 Z) 0                  #  Skip this model in output image?  (yes=1, no=0)
 """

   string_galaxy = """
 0) %s             # object type
 1) %0.2f %0.2f  1 1  # position x, y        [pixel]
 3) %0.2f     1        # total magnitude
 4) %0.2f     1        # effective radius
 5) %0.2f      1        # index
 9) %0.2f      1        # axis ratio
 10) %0.2f    1        # PA
 Z) 0                  #  Skip this model in output image?  (yes=1, no=0)
 """

   string_init_con = """
# Component/    parameter   constraint  Comment
# operation    (see below)    range
"""

   string_psf_con = """
# Component %i: psf
      %i           x          -2 2      #
      %i           y          -2 2      #
      %i           3          -5 5      # mag
"""

   string_gal_con = """
# Component %i: %s
      %i           x          -2 2      #
      %i           y          -2 2      #
      %i           3          -5 5      # mag
      %i           4          5 to 50   # length
      %i           5        0.3 to 8    # index
      %i           9        0.1 to 1    # axis ratio
"""

   string_components = ""
   string_constraint = ""
   for b in range(len(comps)):
      c = b + 2
      if len(comps[b]) == 3:
         string_components += string_psf % (comps[b][0], comps[b][1], 
                                            comps[b][2])
         string_constraint += string_psf_con % (c, c, c, c)
      elif len(comps[b]) == 8:
         string_components += string_galaxy % (comps[b][0], comps[b][1],
                                               comps[b][2], comps[b][3], 
                                               comps[b][4], comps[b][5], 
                                               comps[b][6], comps[b][7])
         string_constraint += string_gal_con % (c, comps[b][0], c, c, c, c, c, 
                                                c)

   f = open("%s/input" % outputdir,"w")
   f.write(string_init % (fsf, img_size[0], img_size[1], img_size[0],
                          img_size[1], zp, img_scale[0], img_scale[1]))
   f.write(string_sky % sky)
   f.write(string_components)
   f.close()

   f = open("%s/constraints" % outputdir,"w")
   f.write(string_init_con)
   f.write(string_constraint)
   f.close()
   return


def input_to_guess(filename):
   """
   Reads in a GALFITM input file ``filename`` and returns an array for use as
   "comps" in make_galfitm_input(). This is only helpful if you have your best
   guesses (perhaps from manual tinkering) in an input file.

   Parameters
   ----------
   filename : TYPE
      File path of desired GALFITM input file.

   Returns
   -------
   comps : list
      Array of the initial parameter guesses for each component in input file.

   """
   comps = []
   with open(filename, 'r') as inpt:
      lines = [line.rstrip() for line in inpt]
      readingGal = False
      readingPSF = False
      comp = []
      for line in lines:
         if len(line) < 2:
            continue
         if readingPSF:
            words = line.split()
            info = ['1', '2']
            if line[1] in info:
               comp.append(float(words[1]))
               continue
            elif line[1] == '3':
               comp.append(float(words[1]))
               comps.append(comp)
               comp = []
               readingPSF = False
               continue
         if readingGal:
            words = line.split()
            info = ['1', '2', '3', '4', '5', '9']
            if line[1:3] == '10' or line[0:2] == '10':
               comp.append(float(words[1]))
               comps.append(comp)
               comp = []
               readingGal = False
               continue
            elif line[1] in info:
               comp.append(float(words[1]))
               continue
         if line[1] == '0' and line[0] != '1':
            if line[4:7] == 'psf':
               readingPSF = True
               continue
            elif line[4:7] == 'sky':
               continue
            else:
               words = line.split()
               readingGal = True
               comp.append(words[1])
               continue
   return comps


def get_guesses(inputdir='.', double=False, masking=False):
   """
   Interprets the file "input.fits" to produce best guesses expected by
   make_galfitm_input().  Does not return img_scale, and the sky value returned
   is from getBlkgrd() with default parameters

   Parameters
   ----------
   inputdir : string, optional
      Path to folder containing desired input FITS file. The default is '.'.
      
   double : bool, optional
      Toggle - if True, then the flux of the central source will be split
      between a central PSF and a sersic of appropriate size. The default is
      False.
      
   masking : bool, optional
      Toggle - if True, then any source not touching the central source will be
      masked, and a file called 'mask.fits' will be created for use with
      GALFITM in the given directory. The default is False.

   Returns
   -------
   size : tuple
      Pixel size of input image.
   
   zp : float
      Photometric zeropoint as derived from image header.
   
   sky : float
      Background level from getBkgrd().
   
   comps : list
      Complex list of estimates expected by make_galfitm_input().
   """
   image = '%s/image.fits' % inputdir
   data = dataPull(image) 
   hdu = fits.open(image)
   size = tuple(hdu[0].data.shape)
   hdr = hdu[0].header
   zp = -2.5*np.log10(hdr["PHOTFLAM"]) + hdr["PHOTZPT"]
   sky = getBkgrd(image)
   data_sub = data - sky
   # ====== CONVOLUTION ========
   kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
   data_conv = convolve(data_sub, kernel)
   data_sub = data_conv # this is bad notation, but I'm adding convolution
                        # after the fact and want it to be easily removed
                        # without affecting the overall code
   # ===========================
   npixels = 300 # change to 50 w/o convolution
   sigma_clip = SigmaClip(sigma=3.5, maxiters=10) # change sigma=3 w/o convol.
   threshold = detect_threshold(data_sub, nsigma=3.0, sigma_clip=sigma_clip)
   segment_img = detect_sources(data_sub, threshold, npixels=npixels)
   
   # Masking of non-overlapping sources (twice, to elim. proximal sources)
   center = np.asarray(size) / 2
   if masking:
      mask = get_mask(data_sub, segment_img, center)
      
      threshold = detect_threshold(data_sub, nsigma=3.0, sigma_clip=sigma_clip,
                                   mask=mask)
      segment_img = detect_sources(data_sub, threshold, npixels=npixels, 
                                   mask=mask)
      mask2 = get_mask(data_sub, segment_img, center)
      mask = np.add(mask, mask2)
      write_mask(mask, inputdir)
   
   # Recreate segmentation image with mask, and deblend
      threshold = detect_threshold(data_sub, nsigma=3.0, sigma_clip=sigma_clip,
                                mask=mask)
      segment_img = detect_sources(data_sub, threshold, npixels=npixels,
                                   mask=mask)
   segment_deb = deblend_sources(data_sub, segment_img, npixels=npixels, 
                                 nlevels=32, contrast=0.02, progress_bar=False)

   # Sort table of values by proximity to center
   cat = SourceCatalog(data_sub, segment_deb)
   cat.fluxfrac_radius(0.5, name='r_e')
   tab = cat.to_table(['xcentroid', 'ycentroid', 'r_e', 'kron_flux', 
                       'elongation'])
   tab['r_e'] = tab['r_e'].value # remove units
   tab['ellip'] = tab['elongation'].value ** -1.
   # Note to self: deblending doesn't seem to be of much use as 6 gets very
   #               wonky, and has overlapping deblends
   
   tab['mag'] = -2.5 * np.log10(tab['kron_flux'] / hdr['EXPTIME']) + zp
   del tab['kron_flux']
   
   # setting a bunch of guesses to a single value as they're too hard to est.
   tab['index'] = 2.
   tab['PA'] = 0.
   
   # Sorting the table by distance to center
   locs = np.zeros((len(tab), 2))
   locs[:,0], locs[:,1] = tab['xcentroid'], tab['ycentroid']
   tab['dist'] = np.linalg.norm(locs - center, axis=1)
   tab.sort('dist')
   
   tab = tab['xcentroid', 'ycentroid', 'mag', 'r_e', 'index', 'ellip', 
             'PA']
   if double:
      tab['mag'][0] = tab['mag'][0] + 0.3
      psfco = [tab['xcentroid'][0], tab['ycentroid'][0], tab['mag'][0]]
      psfco = [round(elem, 2) for elem in psfco]
      comps = [psfco]
   else:
      comps = []
   for s in range(len(tab)):
      comp = ['sersic'] + [ round(elem, 2) for elem in list(tab[s]) ]
      comps.append(comp)
   return size, zp, sky, comps


def get_mask(data, seg_im, coords):
   """
   Produces a mask of given ``data`` to remove sources not at given
   coordinates.

   Parameters
   ----------
   data : string
      File path to input data.
      
   seg_im : segmentation image (array-like)
      Segmentation map produced from ``data``.
      
   coords : list
      List of target centroids not to omit - the source with the centroid 
      closest to each of these coordinates will not be masked.

   Returns
   -------
   mask : array
      Mask as described.

   """
   cat = SourceCatalog(data, seg_im)
   tab = cat.to_table(['xcentroid', 'ycentroid', 'bbox_xmin', 'bbox_xmax', 
                       'bbox_ymin', 'bbox_ymax'])
   if coords.shape == (2,):
      coords = np.asarray([coords])
   for coord in coords:
      locs = np.zeros((len(tab), 2))
      locs[:,0], locs[:,1] = tab['xcentroid'], tab['ycentroid']
      tab['dist'] = np.linalg.norm(locs - coord, axis=1)
      tab.sort('dist')
      tab = tab[1:]
   tab = tab['bbox_xmin', 'bbox_xmax', 'bbox_ymin', 'bbox_ymax']
   mask = np.zeros(data.shape, dtype=bool)
   for n in range(len(tab)):
      reg = list(tab[n])
      mask[reg[2]:reg[3]+1, reg[0]:reg[1]+1] = True
   return mask


def write_mask(mask, inputdir="."):
   """
   Writes given boolean ``mask`` to a binary FITS file in given directory
   ``inputdir``

   Parameters
   ----------
   mask : array-like
      Numpy boolean mask to send to FITS file.
      
   inputdir : string, optional
      Directory where to save output "mask.fits". The default is ".".

   Returns
   -------
   None.

   """
   image = '%s/image.fits' % inputdir
   
   # Load the image and the WCS
   hdul = fits.open(image)
   #wcs = WCS(hdul[1].header)

   # Put the cutout image in the FITS HDU
   hdul_new = fits.PrimaryHDU(mask.astype(int))

   # Update the FITS header with the cutout WCS
   hdr_new = hdul_new.header
    
   # Copy header
   hdr_new = hdul[0].header
   #hdr_new.update(wcs.to_header())

   # Write the cutout to a new FITS file
   hdul_new.writeto('%s/mask.fits' % inputdir, overwrite=True)
   return


def estimate_outer_sky(img, frac=0.25, mask=None):
   ny, nx = img.shape
   y0, y1 = int(ny*(1-frac)), ny
   x0, x1 = int(nx*(1-frac)), nx
   patch = img[y0:y1, x0:x1]
   if mask is not None:
      mpatch = mask[y0:y1, x0:x1]
      patch = patch[~mpatch]
   return np.nanmedian(patch)


def disp_galfitm(inputdir='.', outputdir='.', save=True, name='galim.png',
                thorough=False, masking=False, psfsub=False, radprof=False,
                scale=0.0, errmap=None, **kwargs):
   """
   Takes in GALFITM output files to produce an adjustable summary figure - 
   capable of displaying data, fits, fit components, residuals, and radial
   profiles.

   Parameters
   ----------
   inputdir : Tstr, optional
      File location where to draw files (i.e. where GALFITM was run). The
      default is '.'.
   
   outputdir : str, optional
      File location where to save image. The default is '.'.
   
   save : bool, optional
      Toggles saving figure (if False, just displays). The default is True.
   
   name : str, optional
      File name for output image. The default is 'galim.png'.
   
   thorough : bool, optional
      Toggle for whether or not to display fit components. The default is 
      False, which does not display those components.
      Note: Currently not functional
   
   masking : bool, optional
      Toggle whether to use a mask on model and residual plots. If True, uses 
      'mask.fits' in ``inputdir``. The default is False.
   
   psfsub : bool, optional
      Toggle whether to display a PSF-subtracted panel, i.e. data with the
      centralmost PSF subtracted, as is useful for AGN. The default is False.
   
   radprof : bool, optional
      Toggle for radial profile panel. The default is False.
   
   scale : float, optional
      Pixel scale in arcsec for displaying 5" scalebar. The default is 0.0,
      which does not display a bar.
   
   errmap : array-like, optional
      Error map with the same shape as the data, for use in scaling the
      residual plot. The default is None, which scales based on image standard
      deviation.
   
   **kwargs
      Passed to rp_plot.

   Returns
   -------
   None.

   """
   # Get data from file
   file = '%s/galfitm.fits' % inputdir
   with fits.open(file) as hdul:
      ext_list = [hdu.name for hdu in hdul]
      main_exts = ['INPUT', 'MODEL', 'RESIDUAL']
      input_data = hdul['INPUT'].data
      model_data = hdul['MODEL'].data
      res_data = hdul['RESIDUAL'].data
      if thorough or radprof:
         comp_exts = [s for s in ext_list if 'COMPONENT' in s and not 'sky' \
                      in s]
         compnum = len(comp_exts)
         comp_datas = []
         for comp_ext in comp_exts:
            comp_datas.append(hdul[comp_ext].data)
         comp_dict = dict(zip(comp_exts, comp_datas))
   
   if masking:
      mask = np.ma.make_mask(dataPull('%s/mask.fits' % inputdir))
      if (str(type(mask)) == "<class 'numpy.bool_'>" or
          str(type(mask)) == "<class 'numpy.bool'>"):
         mask = None
   else:
      mask = None
   cols = 3
   if radprof:
      cols += 1
   if psfsub:
      cols += 1
   arrange = 1+math.ceil(compnum/cols)
   if thorough:
      fig = plt.figure(figsize=(6*cols, 6 * arrange))
   else:
      fig = plt.figure(figsize=(6*cols, 6))
   hr, p = [4,1], arrange-1
   while p>0:
      hr.append(5)
      p-=1
   sp = 0.05 # spacing factor : hspace, wspace
   if thorough:
      gs = GridSpec(arrange+1, cols, wspace=sp, hspace=sp, height_ratios=hr)
   else:
      gs = GridSpec(2, cols, wspace=sp, hspace=sp, height_ratios=hr[0:2])
   
   # Process and clean data
   bkgrd = getBkgrd(input_data)
   llim1, ulim1 = np.percentile(input_data, [1, 99])
   norm = ImageNormalize(stretch=LogStretch(), vmin=bkgrd, vmax=ulim1)
   # generate psf-subtracted image, if needed
   if psfsub:
      psf1_ext = next(x for x in comp_exts if 'psf' in x)
      psfsub_data = input_data - comp_dict[psf1_ext]
      psfsub_data_cut = percentile_cut(psfsub_data, 1, 99, mask=mask)
      psfsub_data_cut = np.ma.masked_array(psfsub_data_cut, mask=mask)
   # generate errmap, if needed
   llim2, ulim2 = np.percentile(res_data, [1, 99])
   if errmap is None:
      noise = np.std(res_data[(res_data>llim2) & (res_data<ulim2)])
   else:
      noise = errmap
   res_data = np.where(mask==0, res_data, 0)
   # percentile cuts
   input_data_cut = percentile_cut(input_data, 1, 99, mask=mask)
   model_data_cut = percentile_cut(model_data, 1, 99, mask=mask)
   # apply mask
   model_data_cut = np.ma.masked_array(model_data_cut, mask=mask)
   res_data = np.ma.masked_array(res_data, mask=mask)
   
   if psfsub:
      titles = ['Data', 'Model', 'Data - PSF', 'Residual']
      datas = [input_data_cut, model_data_cut, psfsub_data_cut, res_data]
   else:
      titles = ['Data', 'Model', 'Residual']
      datas = [input_data_cut, model_data_cut, res_data]
   
   # Populate axes
   origin = 'lower'
   cmap = 'viridis'
   interpolation = 'nearest'
   for n in range(len(titles)):
      ax = fig.add_subplot(gs[0:2, n])
      ax.set_axis_off()
      ax.grid()
      if scale > 0:
         corn = len(datas[n]) / 20
         pts = [[corn, corn+(5/scale)], [corn, corn]]
         ax.plot(pts[0], pts[1], 'w-', linewidth=4)
         ax.text(corn+(2.4/scale), corn * (6.5/5), '5"', color='w', 
                 fontweight='bold', fontsize='large')
      ax.set_title(titles[n])
      if n < 3:
         ax.imshow(datas[n], norm=norm, origin=origin, cmap=cmap, 
                   interpolation=interpolation)
      else:
         im = ax.imshow(datas[n]/noise, origin=origin, vmin=-3, vmax=5,
                        cmap=cmap, interpolation=interpolation)
         divider = make_axes_locatable(ax)
         cax = divider.append_axes('bottom', size='4%', pad=0.02)
         cbar = fig.colorbar(im, cax=cax,orientation='horizontal')
         cbar.ax.tick_params(labelsize=14, width=1.5)
         cbar.set_label(label = r"$\sigma$", fontsize=14, labelpad=-4)
   
   # Radial Profile
   if radprof:
      locators = [0, 1, cols-1]
      rp_plot(fig, gs, locators, inputdir=inputdir, save=False, mask=mask,
              **kwargs)
   # Subcomps
   if thorough:
      for n, ext in enumerate(comp_exts):
         data = comp_dict[ext]
         if masking:
            data = np.ma.masked_array(data, mask=mask)
         data_cut = percentile_cut(data, 1, 99)
         ax = fig.add_subplot(((n-1)//cols)+2, gs[((n-1)%cols)-1])
         ax.set_axis_off()
         ax.grid()
         ax.imshow(data_cut, norm=norm, origin=origin, cmap=cmap, 
                   interpolation=interpolation)
   
   if save:
      plt.savefig('%s/%s' % (outputdir, name), bbox_inches='tight')
   else:
      plt.show()
   plt.close()
   return


def rp_plot(fig, gs, locators, comps=['unknown'], inputdir='.', outputdir='.',
            save=False, name='radprof.png', flx=False, res=True, ylim=30.,
            **kwargs):
   """
   Generates a raidal profile from the most recent GALFITM run in the given
   figure.

   Parameters
   ----------
   fig : matplotlib figure
      Figure wherein to place the radial profile.
      
   gs : GridSpec
      GridSpec wherein to place the radial profile.
      
   locators : list, int
      3 ints giving the starting row, ending row, and single column where in
      ```gs``` to place the radial profile.
      
   comps : list, str, optional
      List of component nams for legend. The default is ['unknown'].
      
   inputdir : str, optional
      Directory where to pull GALFITM data. The default is '.'.
      
   outputdir : str, optional
      Directory where to save just the radial profile if toggled by ```save```.
      The default is '.'.
      
   save : bool, optional
      Boolean toggle to save the radial profile on its own. The default is
      False.
      
   name : str, optional
      Filename to save to if ```save``` is set to True. The default is
      'radprof.png'.
      
   flx : bool, optional
      Boolean toggle passed to rad_from_file(). The default is False.
      
   res : bool, optional
      Boolean toggle to make a residual plot underneath the radial profile.
      The default is True.
      
   ylim : float, optional
      Lower bound on y-axis of radial profile, preventing arbitrarily dim outer
      regions of the model from plotting. The default is 30..
      
   **kwargs
      Passed to rad_from_file()

   Returns
   -------
   None.

   """
   if comps[0] == 'unknown':
      comps = ['data', 'total model', 'PSF', 'sersic', 'contaminant']
   colors = ['k', '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628',
             '#984ea3','#999999', '#e41a1c', '#dede00']
   
   muss, x = rad_prof_from_file(inputdir=inputdir, flx=flx, **kwargs)
   
   # Add to axis
   if res:
      ax1 = fig.add_subplot(gs[locators[0],locators[2]])
      ax2 = fig.add_subplot(gs[locators[1],locators[2]], sharex=ax1)
   else:
      ax1 = fig.add_subplot(gs[locators[0]:locators[1]+1,locators[2]])
   
   for n, mus in enumerate(muss):
      style = 'solid'
      if n < len(comps):
         color = colors[n]
         label = comps[n]
      else:
         if n < len(colors):
            color = colors[n]
         else:
            color = colors[n-len(colors)]
            style = 'dashed'
            if n > 20:
               print("You modelled with more than 16 contaminants, which " +
                     "this plotting set can't handle.  How did you even do" +
                     "that???")
         label = comps[-1]
      ax1.plot(x, mus, color=color, label=label, linestyle=style)
   
   xlab = "Radial Distance [arcsec]"
   ax1.legend() # Must come before residuals
   
   # Residuals
   if res:
      residual = muss[0] - muss[1]
      # Replacing add_residual function
      flat = np.zeros_like(x) 
      ax2.plot(x, flat, color='k', linewidth=1)
      ax2.scatter(x, residual, color="crimson", marker='x')
      ax2.ticklabel_format(axis='y', style='sci')
      ax2.set_xlabel(xlab)
      ax2.set_ylabel(r'$\Delta\mu$')
      ax2.yaxis.set_label_position("right")
      ax2.yaxis.tick_right()
      ax1.set_xlim(left=0, right=x[-1])
      #ax1.set_xticks([])
      ax1.tick_params(axis='x', labelbottom=False)
   else:
      ax1.set_xlabel(xlab)
      ax1.set_xlim(left=0, right=x[-1])
   
   if flx:
      plotutils(ax1, yscale='log')
      ax1.set_ylabel("Flux")
      #ax1.set_ylim(bottom=100)
   else:
      plotutils(ax1)
      ax1.set_ylabel(r"$\mu$ [mag arcsec$^{-2}$]")
      ax1.invert_yaxis()
      #ax1.set_ylim(bottom=ylim) # Toggle
   ax1.yaxis.set_label_position("right")
   ax1.yaxis.tick_right()
   return


def rad_prof_from_file(image='image', file='galfitm', exts=[], sky=None,
                       inputdir='.', loc=(-1,-1), radfrac=1.0, mask=None,
                       **kwargs):
   """
   Wrapper for radial_profile() which pulls data and header information and
   loops through the desired extensions of ```file```.

   Parameters
   ----------
   image : str, optional
      Filename to retrieve header information from, such as WCS. The default is
      'image', and name is appended with '.fits'.
   
   file : str, optional
      Filename to retrieve data from. The default is 'galfitm', and
      name is appended with '.fits'.
      
   exts : list of int or str, optional
      Extension from which to retrieve data. The default is [], which will
      prompt automatic usage of 'INPUT', 'MODEL', and all components except
      sky.
   
   sky : array-like, optional
      Array of sky data to use.  Required if supplying exts, otherwise
      retrieved automatically.
      
   inputdir : str, optional
      Directory to find files. The default is '.'.
      
   loc : tuple, int, optional
      Pixel coordinate where to center the radial profile. The default is
      (-1,-1), which defaults to the image center.
      
   radfrac : float, optional
      Sets the radius r of the radial profile, equal to 1/2 image size times
      ```radfrac```. The default is 1.0.
   
   mask : ndarray, optional
      Optional image mask. The default is None.
      
   **kwargs
      Passed to radial_profile().

   Returns
   -------
   muss : array
      Radial profile as a 2D array where each row represents a profiled
      component, and each column corresponding to the radii below.
   
   radii : list of floats
      Radii, in arcsec, corresponding to the center of the bins of the radial
      profile points above.

   """
   file = '%s/%s.fits' % (inputdir, file)
   image = '%s/%s.fits' % (inputdir, image)
   
   # Get data
   with fits.open(file) as hdul:
      main_exts = ['INPUT', 'MODEL']
      if len(exts) == 0:
         ext_list = [hdu.name for hdu in hdul]
         comp_exts = [s for s in ext_list if 'COMPONENT' in s and not 'sky' \
                      in s]
         sky_ext = [s for s in ext_list if 'COMPONENT' in s and 'sky' in s][0]
         exts = main_exts + comp_exts
      elif sky is None:
         raise ValueError('Must provide sky array if providing extension list')
      
      datas = []
      sky_img = hdul[sky_ext].data.astype(float)
      for ext in exts:
          data = hdul[ext].data.astype(float)
          if ext in main_exts:
              data = data - sky_img
          datas.append(data)
   
   # Get header information from image
   with fits.open(image) as hdu:
      hdr = hdu[0].header
      shp = tuple(hdu[0].data.shape)
   radius = shp[0] * radfrac / 2
   if loc == (-1, -1):
      loc = (shp[1]/2, shp[0]/2)
   xycen = loc
   fscale = hdr['fscale']
   radii = np.linspace(1, radius, num=50)
   zp = -2.5*np.log10(hdr["PHOTFLAM"]) + hdr["PHOTZPT"]
   exptime = hdr['EXPTIME']
   
   # Get radial profiles
   muss = []
   for data in datas:
      mus = radial_profile(data, radii, xycen, fscale, zp,
                           exptime=exptime, mask=mask, **kwargs)
      muss.append(mus)
   
   # Testing checks
   print(f'INPUT checks:\nINPUT[0:10] = {muss[0][0:10]}\nnanmin(INPUT)',
         f'= {np.nanmin(muss[0])}\nnanmedian(INPUT) =',
         f'{np.nanmedian(muss[0])}\nnanmax(INPUT) = {np.nanmax(muss[0])}')
   print(f'MODEL checks:\nMODEL[0:10] = {muss[1][0:10]}\nnanmin(MODEL)',
         f'= {np.nanmin(muss[1])}\nnanmedian(MODEL) =',
         f'{np.nanmedian(muss[1])}\nnanmax(MODEL) = {np.nanmax(muss[1])}')
   
   muss = np.asarray(muss)
   
   return muss, ((radii[1:] + radii[:-1]) / 2) * fscale


def radial_profile(data, radii, xycen, fscale, zp, exptime=1., flx=False,
                   mask=None):
   """
   Produces a radial profile describing the brightness of an image from the
   center outwards.  Inputs are processed by photutils RadialProfile.

   Parameters
   ----------
   data : ndarray
      2-dimensional array of image data to be profiled.
      
   radii : ndarray
      Desired radii at which to take profile.
      
   xycen : tuple of floats
      Pixel coordinate at which to center profile.
      
   fscale : float
      Plate scale in arcsec/pixel.
      
   zp : float
      Photometric zeropoint.
      
   exptime : float, optional
      Optional exposure time if data is in a non-rate unit. The default is 1.
      
   flx : bool, optional
      Toggle to return profile in fluxes or magnitudes / area - by which I mean
      units of [whatever a pixel's units are] per [angular area unit as given
      by the square of fscale] - for example, electrons per arcsec^2 - or that
      quantity converted to a magnitude per [angular area unit as given by the
      square of fscale]. The default is False, which produces magnitudes /
      area.
      
   mask : ndarray, optional
      Optional image mask. The default is None.

   Returns
   -------
   mus: ndarray
      The desired radial profile at the given radii.

   """
   rp = RadialProfile(data, xycen, radii, mask=mask)
   
   if flx:
      mus = rp.profile
   else:
      prof = np.array(rp.profile, dtype=float)
      prof[~np.isfinite(prof)] = np.nan
      prof = prof / (fscale**2)
      mus = np.full_like(prof, np.nan)
      pos = prof > 0
      mus[pos] = -2.5*np.log10(prof[pos]) + zp
   return mus


def latex_tab(inputdir='.'):
   """
   Pulls the fit information from the most recent run (in galfitm.galfit.01)
   and store that as a LaTeX-compatible table.

   Parameters
   ----------
   inputdir : string, optional
      Directory to find fit information. The default is '.'.

   Returns
   -------
   None.

   """
   param_file = '%s/galfitm.galfit.01' % inputdir
   paramss = input_to_guess(param_file)
   names = ['kind', 'x', 'y', 'mag', 'length', 'sersic', 'axis ratio', 'angle']
   mt = ['',-99,-99,-99,-99,-99,-99,-99]
   mts = []
   for params in paramss:
      mts.append(mt)
   mts = np.asarray(mts)
   tab = Table(mts, names=names)
   for n, params in enumerate(paramss):
      if len(params) == 3:
         tab['kind'][n] = 'psf'
         for i in range(3):
            tab[names[i+1]][n] = params[i]
      else:
         for i, param in enumerate(params):
            tab[names[i]][n] = params[i]
   tabname = '%s/fit.tab' % inputdir
   tab.write(tabname, format='latex', overwrite=True)
   with open(tabname, 'r') as file:
      data = file.read()
      data = data.replace("-99", "")
      #file.write(data)
   with open(tabname, 'w') as file:
      file.write(data)
   return


def plotutils(ax, xscale='linear', yscale='linear'):
   """
   Applies my favorite formatting to normal, 2D figures, enabling minor tick
   marks, setting all marks as inward-facing, and enabling right and top marks.
   Can optionally change the x/y scale to, say, log. Credit and curses to
   Rogier Windhorst for ingraining this formatting in my mind.

   Parameters
   ----------
   ax : matplotlib axis
      Desired axis on which to apply changes.
      
   xscale : string, optional
      Desired x-axis scaling. The default is 'linear'.
      
   yscale : string, optional
      Desired y-axis scaling. The default is 'linear'.

   Returns
   -------
   None.

   """
   ax.xaxis.set_minor_locator(AutoMinorLocator())
   ax.yaxis.set_minor_locator(AutoMinorLocator())
   ax.tick_params(which='both', direction='in', right=True, top=True)
   ax.set_xscale(xscale)
   ax.set_yscale(yscale)
   return


def cp_psf(file, todir='.'):
   """
   Copies desired PSF model to working directory, saving it under 'psf.fits'.

   Parameters
   ----------
   file : str
      Full path to the desired PSF model. Must be as relative to 
      ```todir```, if using.
      
   todir : str, optional
      Path to working directory. The default is '.'.

   Returns
   -------
   None.

   """
   os.system("cd %s ; cp %s psf.fits" % (todir, file))
   return


def get_flags(file=None, fromdir='.'):
   """
   Checks for errors produced during the supplied run of GALFITM.

   Parameters
   ----------
   file : str, optional
      File to check for flags. The default is None, which looks for
      galfitm.fits.
      
   fromdir : str, optional
      Directory from which to retrieve file. The default is '.'.

   Returns
   -------
   flags : int
      1 if maximum number of iterations was reached, 2 if there was a numerical
      convergence error, and 0 if neither.

   """ 
   flag = 0
   if file is None:
      filename = '%s/galfitm.fits' % (fromdir)
   else:
      filename = '%s/%s.galfitm.fits' % (fromdir, file)
   with fits.open(filename) as hdul:
      flags = hdul[2].header['FLAGS'].split()
      if '1' in flags:
         flag = 1
      elif '2' in flags:
         flag = 2
   return flags


def get_header_param(param, ext=2, file=None, fromdir='.'):
   """
   Pulls a desired parameter from a given FITS file.

   Parameters
   ----------
   param : str
      Desired header parameter.
      
   ext : int, optional
      Extension to pull from. The default is 1.
      
   file : str, optional
      Desired FITS file. The default is None, which will pull from
      'galfitm.fits'.
      
   fromdir : str, optional
      Directory where to find file. The default is '.'.

   Returns
   -------
   prm
      Header result.

   """
   if file is None:
      filename = '%s/galfitm.fits' % (fromdir)
   else:
      filename = '%s/%s.fits' % (fromdir, file)
   with fits.open(filename) as hdul:
      prm = hdul[ext].header[param]
   return prm


def to_latex(tab, outputdir='.', tabname=None, delval='-99.0'):
   """
   Produces and saves a latex table from given Astropy table, creating blanks
   as desired.

   Parameters
   ----------
   tab : astropy table
      Desired table to pass to a latex file.
      
   outputdir : str, optional
      Directory where to store the file. The default is '.'.
      
   tabname : str, optional
      File name to store latex table. The default is None, storing under
      'fit.tab'.
      
   delval : str, optional
      Value which should be excised from latex table, leaving blanks. The
      default is '-99.0'.

   Returns
   -------
   None.

   """
   if tabname is None:
      tabname = '%s/fit.tab' % outputdir
   else:
      tabname = '%s/%s' % (outputdir, tabname)
   tab = Table(tab, masked=True)
   if delval is not None:
      for cn in tab.colnames:
         #tab[cn].mask = (tab[cn] == "-99") | (tab[cn] == -99)
         tab[cn].mask = (tab[cn] == delval) | (tab[cn] == float(delval))
   tab.write(tabname, format='latex', overwrite=True)
   return


def from_latex(fromdir='.', tabname='fit.tab', delval=None, masked=True):
   """
   Inverse of to_latex, returning an astropy table from given latex table,
   optionally adding a desired value to replace blanks.

   Parameters
   ----------
   fromdir : str, optional
      Directory from which to pull file. The default is '.'.
      
   tabname : str, optional
      Name of table file. The default is 'fit.tab'.
      
   delval : str, optional
      Value to replace blanks. The default is None.
      
   masked : str, optional
      Passed to Astropy Table. The default is True.

   Returns
   -------
   tab : astropy table

   """
   fulltab = '%s/%s' % (fromdir, tabname)
   tab = Table.read(fulltab, format='latex')
   tab = Table(tab, masked=masked, copy=False)
   if delval is not None:
      tab = tab.filled(delval)
   return tab


def percentile_cut(data, lower=None, upper=None, truncate=True, mask=None):
   """
   Does what I wish astropy PercentileInterval did. Returns a copy of the given
   array with values outside bounds removed/truncated.

   Parameters
   ----------
   data : ndarray
      Input array.
      
   lower : float, optional
      Lower percentile bound. The default is None.
      
   upper : float, optional
      Upper percentile bound. The default is None.
      
   truncate : bool, optional
      Toggle behavior to truncate or remove values past upper and lower bounds.
      The default is True, truncating (setting values past the bound to bound).
   
   mask : ndarray, optional
      Boolean mask of pixels to ignore when calculating.

   Returns
   -------
   datat : ndarray
      Altered array

   """
   if mask is None:
      wdata = data.copy()
   else:
      wdata = data[~mask].copy()
   wdata.setflags(write=True)
   if lower is None:
      if upper is None:
         return data
      llim = 1
   else:
      llim = np.nanpercentile(wdata, lower)
   if upper is None:
      ulim = 1
   else:
      ulim = np.nanpercentile(wdata, upper)
   if truncate:
      lims = [llim, ulim]
   else:
      lims = [np.nan, np.nan]
   if mask is None:
      if lower is None:
         datat = np.where(data > ulim, lims[1], data)
      elif upper is None:
         datat = np.where(data < llim, lims[0], data)
      else:
         low_data = np.where(data < llim, lims[0], data)
         datat = np.where(low_data > ulim, lims[1], low_data)
   else:
      if lower is None:
         datat = np.where((data > ulim) & ~mask, lims[1], data)
      elif upper is None:
         datat = np.where((data < llim) & ~mask, lims[0], data)
      else:
         low_data = np.where((data < llim) & ~mask, lims[0], data)
         datat = np.where((low_data > ulim) & ~mask, lims[1], low_data)
   return datat


def errmap_HST(file, box_size=100, purity=0.05, overlap=0.2, nbox_min=20):
   """
   Produces a sigma image by empirically estimating the scaling factor between
   the true SCI background standard deviation and the average value of the WHT
   image.

   Parameters
   ----------
   file : str
      FITS file path.
      
   box_size : float, optional
      Desired estimation box size, in pixels. The default is 100.
      
   purity : float, optional
      Required purity fraction, e.g. how much of a drawn box can be masked
      (covered by a bright source). The default is 0.05.
      
   overlap : float, optional
      Acceptable overlap fraction between drawn boxes. The default is 0.2.
      
   nbox_min : int, optional
      Minimum number of boxes to draw. The default is 20.

   Returns
   -------
   Array sigma image.
   
   Notes
   -----
   The WHT image is assumed to be a linearly scaled inverse variance map, which
   I am confident is the case if final_wht_type='ERR' in HST drizzling, and so
   scaling of values (v) is scaled as sigma = f/sqrt(v), where f is a scaling
   factor described above.
   
   In plain English, the logic here is that there is an unknown scaling factor
   in HST drizzling when producing an error map between the true sigma of a
   given pixel and the inverse-square presented in an error map. If we assume
   this scaling factor is linear, we can approximate it by drawing many
   rectangular regions in background parts of the image and taking the standard
   deviation of the real pixels, and comparing to the average
   inverse-square-root of the error map.  If this scaling factor is 1, the two
   values should be the same; otherwise, it can easily be determined.

   """
   #Get basic data from files  
   image = get_pkg_data_filename(file)
   data_sci = fits.getdata(image, ext=1)
   data_wht = fits.getdata(image, ext=2)
   
   # Mask input data of connected exact zeroes - this is likely to
   # be outside the true image.
   zeros_mask = np.zeros_like(data_sci)
   kernel = np.array([[1, 1, 1],
                      [1, 0, 1],
                      [1, 1, 1]])
   # Find all zero pixels in the image
   zero_pixels = (data_sci == 0)
   # Count zero neighbors for each pixel
   neighbor_zero_count = convolve(zero_pixels.astype(int), kernel,
                                  mode='constant', cval=0)
   selected_pixels = zero_pixels & (neighbor_zero_count >= 2)
   zeros_mask[selected_pixels] = 1
   zeros_mask = zeros_mask.astype(bool)
   
   fsigma = np.sqrt(data_wht) ** -1
   fsigma_clip = percentile_cut(fsigma, 1, 99, mask=zeros_mask) # TODO: Fix
   
   rng = np.random.default_rng()
   
   # Get source mask
   sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
   threshold = detect_threshold(data_sci, nsigma=2.0, sigma_clip=sigma_clip)
   segment_img = detect_sources(data_sci, threshold, npixels=50)
   footprint = circular_footprint(radius=10)
   mask = segment_img.make_source_mask(footprint=footprint)
   
   # Add to source mask the zeros-mask
   mask = mask | zeros_mask
   
   # Draw regions for stats
   overmask = np.zeros_like(mask)
   sigmas = []
   avgs = []
   n_boxes = 0
   iters = 0
   max_iters = 1000
   while n_boxes < nbox_min:
      # Checking for overflow
      if iters > max_iters:
         print("WARNING: Insufficient open sky to produce", str(nbox_min),
               "boxes for statistics; reached", str(max_iters), "iterations.")
         print("Returning statistics based on", n_boxes, "boxes.")
         break
      iters += 1
      
      # Make box
      x = int(rng.random() * (len(mask) - (box_size+1)))
      x = [x, x + box_size]
      y = int(rng.random() * (len(mask[1]) - (box_size+1)))
      y = [y, y + box_size]
      
      # check purity
      if np.sum(mask[x[0]:x[1], y[0]:y[1]]) / (box_size ** 2) > purity:
         continue
      if np.sum(overmask[x[0]:x[1], y[0]:y[1]]) / (box_size ** 2) > overlap:
         continue
      overmask[x[0]:x[1], y[0]:y[1]] = 1
      
      # Compute stats
      sigmas.append(np.std(data_sci[x[0]:x[1], y[0]:y[1]]))
      avgs.append(np.average(fsigma_clip[x[0]:x[1], y[0]:y[1]]))
      
      n_boxes += 1
      # \while
   
   # Collate stats
   sigmas, avgs = np.array(sigmas), np.array(avgs)
   fs = sigmas / avgs
   f = np.nanmean(fs)
   
   return fsigma_clip * f


def errfile(file, outputdir, name="errmap.fits", **kwargs):
   """
   Produces a copy of the given HST file but with the data under the first
   header replaced by an errmap produced by errmap_HST().

   Parameters
   ----------
   file : str
      File path of HST FITS file.
      
   outputdir : str
      Output directory.
      
   name : str, optional
      DDesired file name of the output error map. The default is "errmap.fits".
      
   **kwargs
      Passed to errmap_hst().

   Returns
   -------
   None.

   """
   errmap = errmap_HST(file, **kwargs)
   with fits.open(file) as hdul:
      hdul[1].data = errmap
      hdul.writeto('%s/%s' % (outputdir, name), overwrite=True)
   return


def seek_data(coord, todir, **kwargs):
   '''
   Pulls data as indicated from HST MAST archive, selects appropriate files for
   use with GALFITM, and downloads them to the indicated directory, along with a
   default PSF model.

   Parameters
   ----------
   coords : str
      Coordinates of object, passed to Observations.query_criteria.
   
   todir : str
      Directory to which files should be saved.  This can be the GALFITM
      working directory, but that is not recommended.
   
   **kwargs
      Passed to Observations.query_criteria.

   Returns
   -------
   None.

   '''
   coord_str = coord.to_string('decimal')
   # Identify data available at given coordinate and other given information
   obs_table = Observations.query_criteria(coordinates=coord_str, 
                                           calib_level=3, **kwargs)
   data_products = Observations.get_product_list(obs_table)
   # Use only drizzled products
   driz_products = Observations.filter_products(data_products,
                                                calib_level=[3,4],
                                 productSubGroupDescription="DRZ",
                                 extension="fits", productType='SCIENCE')
   
   # Sort products by the exposure time  of their parent observation
   driz_products = join(driz_products, obs_table['obs_id', 't_exptime'])
   sorted_products = driz_products[np.argsort(driz_products['t_exptime'])[::-1]]
   
   # Only download the top product from above sorting, but check if that
   # product has proper WCS information.
   n = 0
   file_num = len(sorted_products)
   while True:
      best_product = sorted_products[n]["dataURI"]
      name = sorted_products[n]["productFilename"]
      file_path = f'{todir}/{name}'
      result = Observations.download_file(best_product, local_path=file_path)
      try:
         with fits.open(file_path) as hdul:
            hdu = hdul[1]
            scale = [hdu.header['CDELT1'], hdu.header['CDELT2']]
      except KeyError:
         try:
            with fits.open(file_path) as hdul:
               hdu = hdul[1]
               cd1 = [hdu.header['CD1_1'], hdu.header['CD1_2']]
               cd2 = [hdu.header['CD2_1'], hdu.header['CD2_2']]
               scale = [((cd1[0])**2 + (cd2[0])**2)**0.5,
                        ((cd1[1])**2 + (cd2[1])**2)**0.5]
         except KeyError:
            print(f'''The automatically selected best MAST FITS file ({n+1}/
                  {file_num}) lacks proper WCS information.  This file can be 
                  found at:\n{file_path}\n and should be inspected.''')
            n += 1
            if n == file_num:
               raise NoWCSError(f'''No files were found in MAST containing 
                                appropriate WCS information. This is unusual.''')
            else:
               print('Trying next file...')
            continue
      break
   # compare obs_collection, obs_id, project, and proposal_id
   #manifest = Observations.download_products(data_products,
   #                                          download_dir=todir, flat=True,
   #                                          productType="SCIENCE")
   return file_path


def download_psf(filter_name, psf_dir):
   '''
   Checks if the empirical PSF FITS for a given filter exists locally; if not,
   downloads it from STScI.

   Parameters
   ----------
   filter_name : str
      What filter to find a PSF for.  Currently, only WFC3 is supported.
   
   psf_dir : str
      What directory to check for a PSF, and if not found, save one.

   Raises
   ------
   ValueError
      Alerts if filter is not supported.

   Returns
   -------
   filepath : str
      Path to new empirical PSF file.
   
   Notes
   -----
   TODO: Make sure this works, minimally edited from Perplexity

   '''
   base_psf_url = "https://www.stsci.edu/~jayander/HST1PASS/LIB/PSFs/STDPSFs/WFC3"
   uvis_filters = ['F225W', 'F275W', 'F336W', 'F390W', 'F438W', 'F467M',
                   'F555W', 'F606W', 'F775W', 'F814W', 'F850L']
   ir_filters = ['F105W', 'F110W', 'F125W', 'F140W', 'F160W']
   if filter_name in uvis_filters:
      filt = 'UV'
   elif filter_name in ir_filters:
      filt = 'IR'
   else:
      raise ValueError(f"Error: Filter not supported: {filter_name}\nCurrently only WFC3 filters supported")
   
   base_psf_url += f'{filt}/'
   filename = f"STDPSF_WFC3{filt}_{filter_name}.fits"
   filepath = os.path.join(psf_dir, filename)
   
   if not os.path.isfile(filepath):
       url = f"{base_psf_url}{filename}"
       print(f"Downloading empirical PSF from {url}")
       urllib.request.urlretrieve(url, filepath)
   return filepath


def get_psf(fits_filepath, psf_dir, coord):
   '''
   Ideally, this would locate and excise the empirical PSF best suited for the given location in
   the given file as follows:
      
    - Checks FITS header for WFC3 and filter.
    - Tests that the provided skycoord is within FITS bounds.
    - Ensures empirical PSF (library) is present/downloaded.
    - Finds grid PSF closest to given skycoord and writes it as a new FITS
      file.
   
   In reality, we simply grab the central PSF of the empirical PSF for the
   appropriate band.  This is roughly sufficient as the empirical PSF is never
   an excellent fit to given data regardless, so the variation across the chip
    - already not easy to account for in drizzled images - is negligible
    compared to other errors.

   Parameters
   ----------
   fits_filepath : str
      FITS file to be modelled.
   
   psf_dir : str
      Directory to check and store PSF.
   
   coord : SkyCoord
      Location of object to be modelled.

   Raises
   ------
   ValueError
      Checks for appropriate instrument, filter, and coordinate.

   Returns
   -------
   output_path : str
      Path to output PSF.

   '''
   # Open FITS file and read header info
   with fits.open(fits_filepath) as hdul:
      hdr0 = hdul[0].header
      hdr1 = hdul[1].header
      wcs = WCS(hdr1)
      instrument = hdr0.get('INSTRUME', '').strip().upper()
      filter_name = hdr0.get('FILTER', '').strip().upper()
      # Get image size
      naxis1, naxis2 = hdr1.get('NAXIS1'), hdr1.get('NAXIS2')
   
   # Instrument and filter check
   if 'WFC3' not in instrument:
      raise ValueError(f"Error: File instrument is not from WFC3 observation: {instrument}")
   if not filter_name:
      raise ValueError("Error: No FILTER keyword found in FITS header")

   # Check coord coverage (convert to pixel, test within bounds)
   x_pix, y_pix = skycoord_to_pixel(coord, wcs=wcs)
   if not (0 <= x_pix < naxis1 and 0 <= y_pix < naxis2):
      raise ValueError(f"Error: Sky coordinate {coord.to_string('hmsdms')} is outside image coverage.")

   # Ensure empirical PSF FITS for this filter is present (download if necessary)
   psf_filepath = download_psf(filter_name, psf_dir)

   # Load empirical PSF grid and select the data of the central PSF
   # TODO: It would be better to select the PSF closest to the source
   with fits.open(psf_filepath) as psf_hdul:
      data = psf_hdul[0].data
      cen_psf = math.ceil(len(data)/2)
      psf_data = data[cen_psf,:,:]
      print(type(psf_data))
      psf_hdr = psf_hdul[0].header

   # Write new FITS
   #best_psf = fits.PrimaryHDU(data=psf_data, header=psf_hdr)
   output_path = f'{psf_dir}/psf.fits'
   fits.writeto(output_path, psf_data, overwrite=True)
   #best_psf.writeto(outputname)

   print(f"Extracted PSF saved to: {output_path}")
   return output_path



# =============================================================================
# Useful things I tend to copy-paste
# =============================================================================
# Color blindness friendly cycle - good contrast between subsequent colors
# Credit to thivest on GitHub
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
CBcc = CB_color_cycle # shortname

# =============================================================================
# Experimental code starts here
# =============================================================================


def automate(working_dir, coord, filters, size,
             imname='galim.png', backupdir=None, bkpname=None,
             customDir=None, customParam=None, customCons=None, useError=False,
             double=False, center_psf=False, **kwargs):
   '''
   This function is my attempt at writing a general-use automatic pipeline of
   the other functions in this wrapper.  It is not exhaustive, and leaves out a
   number of utilities and assumes a default behavior with no options for
   configuration.  For example, disp_galfitm has a large number of customization
   options that are forced here, as does bkp_galfitm.

   Parameters
   ----------
   working_dir : str
      Path to working directory where temporary files are stored, relative to
      the current directory.
   
   coord : SkyCoord
      Location of target object.
   
   filters : str
      Filter to pull data for. Currently only works with one filter.
   
   size : int or tuple of ints
      Size, in pixels, of cutout to make around target source. Int implies a
      square cutout.
   
   imname : str, optional
      Name of output image. The default is 'galim.png'.
   
   backupdir : str, optional
      Directory to store output files. The default is None.
   
   bkpname : str, optional
      Naming convention for output files - gets prepended. The default is None.
   
   customDir : str, optional
      Directory path to find ```customParam``` and ```customCons```. The
      default is None.
   
   customParam : str, optional
      File name to find custom GALFITM parameters rather than deriving them
      analytically. The default is None.
   
   customCons : str, optional
      File name to find custom GALFITM parameters rather than a default. The
      default is None.
   
   useError : bool, optional
      Toggle to use image error map to scale residuals. The default is False.
   
   double : bool, optional
      Adds a central PSF component - otherwise all components are Sersic
      profiles. This is typical for AGN science. The default is False.
   
   center_psf : bool, optional
      Toggle to center the radial profile on the PSF component, if used
      (toggled by ```double```). The default is False.
   
   **kwargs
      Passed to disp_galfitm().

   Returns
   -------
   None.
   '''
   # I can't manage variable names to save my life.  Convert size if needed.
   path = working_dir
   if len(size)>1:
      size = tuple(size)
   else:
      size = tuple(size[0], size[0])
   
   # Download best data at coordinate, and then appropriate PSF
   file = seek_data(coord, working_dir, radius=".02 deg", filters=filters)
   get_psf(file, working_dir, coord)
   
   # Pull appropriate pixel scale from file
   with fits.open(file) as hdul:
      hdu = hdul[1]
      try:
         scale = [hdu.header['CDELT1'], hdu.header['CDELT2']]
      except KeyError:
         cd1 = [hdu.header['CD1_1'], hdu.header['CD1_2']]
         cd2 = [hdu.header['CD2_1'], hdu.header['CD2_2']]
         scale = [((cd1[0])**2 + (cd2[0])**2)**0.5,
                  ((cd1[1])**2 + (cd2[1])**2)**0.5]
         scale = [i*3600 for i in scale] #converts from deg/pix to arcsec/pix
   
   # Cut stamp, and appropriate error stamp if toggled.
   cut(file, coord, size, outputdir=path)
   if useError:
      errfile(file, path)
      cut('%s/errmap.fits' % path, coord, size, outputdir=path,
          outputname='errmap_cut.fits')
      errmap = dataPull('%s/errmap_cut.fits' % path)
   else:
      errmap = None
   
   # Generate quick guesses of parameters for GALFITM, and make param file.
   # Also makes a mask file.
   size, zp, sky, comps = get_guesses(path, double=double, masking=True)
   # overwrite param file with custom file if present
   if customDir is not None:
      custom_param_file = '%s/%s' % (customDir, customParam)
      comps = input_to_guess(custom_param_file)
   
   # Generate GALFITM input and constraint files
   make_galfitm_input(size, zp, scale, sky, comps, outputdir=path)
   # overwrite constraint file with custom constraints if present
   if customCons is not None:
      custom_cons_file = '%s/%s' % (customDir, customCons)
      os.system('cp %s %s/constraints' % (custom_cons_file, working_dir))
   run_galfitm(outputdir=path)
   # If using a central PSF, check to center the radial profile on it.
   loc = (-1, -1)
   if double and center_psf:
      comps = input_to_guess(path + '/galfitm.galfit.01')
      psf = next(comp for comp in comps if len(comp) == 3)
      loc = (psf[0], psf[1])
   # Produce output image.
   disp_galfitm(inputdir=path, outputdir=path, save=False, name=imname,
               scale=scale[0], errmap=errmap, loc=loc, **kwargs)
   
   # Back up relevant files.
   if backupdir is not None:
      bkp_galfitm(backupdir, fromdir=path, name=bkpname, gal=True, inpt=False,
                  log=False, image=imname)
   return


# =============================================================================
# Scrap code begins here
# =============================================================================
filters = 'F160W' #F160W
working_dir = 'testing'
size = (200, 200)
coord = SkyCoord(212.5857, 36.723, frame='icrs', unit='deg')
# 212.5857, 36.723 is my best AGN with spiral arms, but produces an odd fig
automate(working_dir, coord, filters, size, backupdir='backup',
         bkpname='test', useError=True, double=True, center_psf=True, 
         masking=True, psfsub=True, radprof=True, flx=False, radfrac=0.3)
