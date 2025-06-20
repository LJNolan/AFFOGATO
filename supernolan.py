#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 17:09:39 2024

Author: Liam Nolan

A collection of utilities that I copy-pasted for a while before realizing I
should put them in one file.
"""
from astropy.stats import SigmaClip
from astropy.table import Table
from math import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd
from photutils.aperture import CircularAperture
from photutils.segmentation import detect_threshold, detect_sources
from photutils.utils import circular_footprint


def plotutils(ax, xscale='linear', yscale='linear'):
# =============================================================================
# This function applies my favorite formatting to normal, 2D figures on a given
# matplotlib Axes object (ax). It enables minor tick marks, sets all marks as
# inward-facing, and enables right and top marks.  It can optionally change the
# x and y scale to, say, log, with a given string. Credit and curses to Rogier
# Windhorst for ingraining this formatting in my mind.
# =============================================================================
   ax.xaxis.set_minor_locator(AutoMinorLocator())
   ax.yaxis.set_minor_locator(AutoMinorLocator())
   ax.tick_params(which='both', direction='in', right=True, top=True)
   ax.set_xscale(xscale)
   ax.set_yscale(yscale)
   return


def flatten(xss):
   """
   Turns a list of lists into a flat list.

   Parameters
   ----------
   xss : list
      List to flatten.

   Returns
   -------
   list
      Flattened list.

   """
   return [x for xs in xss for x in xs]


def repeat(x, n):
   """
   Returns a list which is ``n`` loops of ``x``.

   Parameters
   ----------
   x : list
      List to be looped.
   n : int
      Number of repeats.

   Returns
   -------
   list
      Looped list.
   
   Example
   -------
   input : repeat([a, b, c], 3)
   output : [a, b, c, a, b, c, a, b, c]

   """
   return flatten([x for _ in range(n)])


def inv_repeat(x, n):
   """
   Returns a list which has each element of ``x`` repeated ``n`` times.
   Intended as a rough inverse of repeat().

   Parameters
   ----------
   x : list
      List to be expanded.
   n : int
      Number of repeats.

   Returns
   -------
   list
      Expanded list.
   
   Example
   -------
   input : repeat([a, b, c], 3)
   output : [a, a, a, b, b, b, c, c, c]

   """
   return flatten([[e]*n for e in x])


def style_dict(inputs, color_cycle=None, kind_cycle=None, kind=None,
               match=False, lead_color=True):
   """
   Generates two dictionaries that map the values from ``inputs`` to
   combinations of colors and line styles.

   Parameters
   ----------
   inputs : iterable (string)
      The values which are to be mapped to colors and styles.
   
   color_cycle : iterable (string or other color), optional
      Colors to cycle through. The default is None, which uses a
      colorblindness-friendly cycle from thivest on GitHub.
   
   kind_cycle : iterable (string or other linestyle), optional
      Line, hatch, etc. styles to cycle through. The default is None, which
      uses a default list of styles in matplotlib.
      
   kind : str, optional
      Can be 'line' or 'hatch'.  Indicates if the default output should be a
      line style or hatch style, as used for histograms.  Default is None,
      which assumes 'line' is desired if kind_cycle was not provided.
   
   match : bool, optional
      Toggle to Couple or uncouple cycling kind and color - True means to cycle
      together, so each color maps 1:1 to a style (i.e. for colorblind
      accessibility).  The default is False.
   
   lead_color : bool, optional
      Toggle to start cycling by color; if not, cycle by kind style. E.g. make
      a solid line of color 1, then solid of color 2, etc., or make a solid
      line of color 1, then dashed line of color 1, etc. The default is True.

   Returns
   -------
   color_dict : dict
      Dictionary mapping ``input`` to appropriate colors.
   
   kind_dict : dict
      Dictionary mapping ``input`` to appropriate styles.

   """
   if color_cycle is None:
      color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                        '#f781bf', '#a65628', '#984ea3',
                        '#999999', '#e41a1c', '#dede00']
   if kind_cycle is None:
      if kind == 'hatch':
         kind_cycle = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
      else:
         kind_cycle = ['solid', 'dotted', 'dashed', 'dashdot']
   
   longcon = (len(inputs) > len(color_cycle)) or (len(inputs) 
                                                  > len(kind_cycle))
   if len(inputs) > (len(color_cycle) * len(kind_cycle)) or (match 
                                                             and longcon):
      print('ERROR: input too long for given color and line cycles.')
      return
   if match:
      cc = color_cycle
      lc = kind_cycle
   elif lead_color:
      cc = repeat(color_cycle, len(kind_cycle))
      lc = inv_repeat(kind_cycle, len(color_cycle))
   else:
      cc = inv_repeat(color_cycle, len(kind_cycle))
      lc = repeat(kind_cycle, len(color_cycle))
   
   color_dict = dict(zip(inputs, cc[:len(inputs)]))
   kind_dict = dict(zip(inputs, lc[:len(inputs)]))
   
   return color_dict, kind_dict


def squarify(n):
   '''
   Create the smallest square-like grid that can contain ``n`` points.

   Parameters
   ----------
   n : int
      Number of desired points in grid.

   Returns
   -------
   i, i : int, int
      Number of rows, columns for desired grid.  Always has i columns and
      either i or i-1 rows.

   '''
   i = 1
   while i**2 < n:
      i += 1
   i2 = i**2
   if n <= (i2 - i):
      return i-1, i
   else:
      return i, i


def square_subplots(n, **kwargs):
   '''
   Wrapper for plt.subplots to generate a square-like set of axes sufficient
   for ``n`` subplots.

   Parameters
   ----------
   n : int
      DESCRIPTION.
   
   **kwargs : TYPE
      DESCRIPTION.

   Returns
   -------
   fig : matplotlib figure
   
   axes : matplotlib axes
   '''
   return plt.subplots(squarify(n), **kwargs)


def sqsc_subplots(n, scale=6, **kwargs):
   '''
   Wrapper for plt.subplots to generate a square-like set of axes sufficient
   for ``n`` subplots, with a bonus built-in which sizes the figure to a scale
   automatically.

   Parameters
   ----------
   n : int
      DESCRIPTION.
   
   scale: float, optional
      Desired size scale. Default is 6.
   
   **kwargs : TYPE
      DESCRIPTION.

   Returns
   -------
   fig : matplotlib figure
   
   axes : matplotlib axes

   '''
   r, c = squarify(n)
   return plt.subplots(r, c, figsize=(c*scale, r*scale), **kwargs)


def detec_limit(data, fwhm, scale, photflam, photzpt, exptime=None, nsigma=5,
                overlap=0.2, n_ap_min=100, smart=True, ap_max=1e6):
   
   rng = np.random.default_rng()
   ap_area = (pi * ((fwhm/2) ** 2))
   
   # Get source mask
   sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
   threshold = detect_threshold(data, nsigma=2.0, sigma_clip=sigma_clip)
   segment_img = detect_sources(data, threshold, npixels=50)
   footprint = circular_footprint(radius=10)
   mask = segment_img.make_source_mask(footprint=footprint) + np.isnan(data)
   
   # === Smart Aperture Generation === #
   if smart:
      crowds = []
      for n in range(10):
         centers = np.random.uniform(fwhm/2, len(mask)-(fwhm/2), size=(100, 2))
         apers = CircularAperture(centers, fwhm/2)
         over = apers.area_overlap(data, mask=mask) / ap_area
         crowds.append(len(over[over > (1-overlap)]))
      crowds = np.array(crowds)
      inv_crowdedness = np.average(crowds)
      if inv_crowdedness < 1:
         print("WARNING: Image is over 99% crowded. This means less than 1 in",
               "100 apertures of given FWHM successfully plot in unoccupied",
               "background with an overlap fraction of less than ",
               str(overlap), ".")
         inv_crowdedness = 1
      nape = int(n_ap_min * 100 / inv_crowdedness)
      nape += nape // 3 # addl. apertures to reduce impact of randomness
      if nape > ap_max:
         print("WARNING: Smart aperture generation suggests more than allowed",
               "maximum apertures. Consider increasing maximum apertures, or",
               "decreasing minimum apertures.")
         nape = ap_max
      
   else:
      nape = ap_max
   # ================================= #
   
   # Draw regions for stats, confirm meeting minimum apertures
   b = 0
   while b < 4:
      if b < 3:
         centers = np.random.uniform(fwhm/2, len(mask)-(fwhm/2), size=(nape, 2))
         apers = CircularAperture(centers, fwhm/2)
         over = apers.area_overlap(data, mask=mask) / ap_area
         if len(over[over > (1-overlap)]) > n_ap_min:
            break
      elif len(over[over > (1-overlap)]) == 0:
         return
      else:
         print("WARNING: Failed to reach minimum apertures after 3 attempts",
               "Returning depth with existing apertures.")
      b += 1
   
   sums, sum_errs = apers.do_photometry(data, mask=mask)
   
   # Eliminate excessively masked apertures
   sums = sums[over > (1-overlap)]
   over = over[over > (1-overlap)]
   
   # Rescale acceptably-masked apertures
   sums = sums / over
   
   # Generate flux threshold
   sigma = np.std(sums)
   flux = sigma * nsigma / (pi * (((fwhm/2) * scale) ** 2))
   
   limit = (-2.5 * np.log10(flux * photflam / exptime)) + photzpt
   
   return limit


def round_table_sf(table, columns, sigfigs):
   """
   Round specified column(s) in an Astropy Table or Pandas DataFrame to the
   given number(s) of significant digits. (for decimal rounding, see
   round_table_dec()).
    
   Parameters:
   -----------
   table : astropy.table.Table
       Input table.
   
   columns : str or list of str
       Column name(s) to round.
    
   sigfigs : int or list of int
       Number of significant digits for each column. Can be a single int or a
       list matching the columns.
        
   Returns:
   --------
   new_table : astropy.table.Table
       A copy of the table with the specified columns rounded.
   """
   def round_sig(x, sig):
      if x == 0:
         return 0
      return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)
   
   is_astropy = isinstance(table, Table)
   is_pandas = isinstance(table, pd.DataFrame)

   if not (is_astropy or is_pandas):
      raise TypeError("Input must be an Astropy Table or Pandas DataFrame.")
   
   # Normalize inputs
   if isinstance(columns, str):
      columns = [columns]
   if isinstance(sigfigs, int):
      sigfigs = [sigfigs] * len(columns)
   elif len(sigfigs) != len(columns):
      raise ValueError("Length of 'sigfigs' must match length of 'columns'")

   new_table = table.copy()

   for col, sig in zip(columns, sigfigs):
      if (is_astropy and col not in new_table.colnames) or (is_pandas and col not in new_table.columns):
         raise KeyError(f"Column '{col}' not found in the table.")
   
      if is_astropy:
         new_table[col] = [round_sig(val, sig) for val in new_table[col]]
      elif is_pandas:
         new_table[col] = new_table[col].apply(lambda x: round_sig(x, sig))

   return new_table


def round_table_dec(table, columns, decimals):
   """
   Round specified columns in an Astropy Table or Pandas DataFrame to the given
   number of decimal places. (for significant figure rounding, see
   round_table_sf()).

   Parameters:
   -----------
   table : astropy.table.Table
       Input table.
   
   columns : str or list of str
       Column name(s) to round.
   
   decimals : int or list of int
       Number of decimal places for each column. Can be a single int or a list matching the columns.

   Returns:
   --------
   new_table : astropy.table.Table
       A copy of the table with the specified columns rounded.
   """
   is_astropy = isinstance(table, Table)
   is_pandas = isinstance(table, pd.DataFrame)

   if not (is_astropy or is_pandas):
      raise TypeError("Input must be an Astropy Table or Pandas DataFrame.")
   
   # Normalize inputs
   if isinstance(columns, str):
      columns = [columns]
   if isinstance(decimals, int):
      decimals = [decimals] * len(columns)
   elif len(decimals) != len(columns):
      raise ValueError("Length of 'decimals' must match length of 'columns'")

   new_table = table.copy()

   for col, dec in zip(columns, decimals):
      if (is_astropy and col not in new_table.colnames) or (is_pandas and col not in new_table.columns):
           raise KeyError(f"Column '{col}' not found in the table.")

      if is_astropy:
          new_table[col] = [round(val, dec) if np.isfinite(val) else val for val in new_table[col]]
      elif is_pandas:
          new_table[col] = new_table[col].apply(lambda x: round(x, dec) if pd.notnull(x) else x)

   return new_table


# =============================================================================
# Useful things I tend to copy-paste
# =============================================================================
# Color blindness friendly cycle - good contrast between subsequent colors
# Credit to thivest on GitHub
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
CBcc = CB_color_cycle # shortname
