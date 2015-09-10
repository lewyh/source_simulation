from __future__ import print_function, division
__author__ = 'hywel.farnhill'

from scipy.optimize import curve_fit
from astropy.table import Table
from astropy.io import fits
import pkg_resources
import numpy as np
import subprocess
import logging
import urllib
import ntpath
import warnings
import sys
import os


class Field():
    # TODO: Make sure these are UNIX friendly!!
    tempoutdir = '/tmp/simsource/'
    #outdir = '~/simsource/'
    outdir = os.path.expanduser('~') + '/simsource/'

    def __init__(self, _filename, _confmap, _runid, magbins, N, M, dist, magdist, ccd=4, web=False):
        """docstring for __init__"""
        self.filename = _filename
        self.confmap = _confmap
        self.runid = _runid
        self.magbins = magbins

        # Add an extra value to the magnitude bins array so that each bins' range is encompassed by the values
        diff = magbins[-1] - magbins[-2]
        self.magbins = np.append(self.magbins, self.magbins[-1] + diff)

        self.N = N
        self.M = M


        self.dist = dist
        self.magdist = magdist

        self.ccd = ccd

        self.tmp_dir = ''
        self.conffn, self.imgfn = self.setup_files(web)

        self.fits = fits.open(self.imgfn)
        self.img = fits.getdata(self.imgfn, 1)
        self.dim = np.shape(self.img)

        self.parameters = {}

        self.imcorefn = self.imcore(self.imgfn, True)
        stiltsfn = self.stilts(self.imcorefn)
        self.combine(self.imcorefn, stiltsfn)
        self.addmags(self.imcorefn)

        self.cat = fits.open(self.imcorefn)
        self.mask = self.cat[1].data['Classification'] == -1

        self.mag = self.cat[1].data['Mag'][self.mask]
        self.peak = self.cat[1].data['Peak_height'][self.mask]
        self.flux = self.cat[1].data['Core2_flux'][self.mask]

        self.magpeak()
        self.magflux()
        self.posang()
        self.gethead()

        self.imcorefn = self.imcorefn.split('ccd4')

    def setup_files(self, web):
        filename = ntpath.basename(self.filename)
        if filename.endswith('.fits.fz'):
            filename_noext = filename.rstrip('.fits.fz')
        elif filename.endswith('.fits'):
            filename_noext = filename.rstrip('.fits')
        elif filename.endswith('.fit'):
            filename_noext = filename.rstrip('.fit')

        if not os.path.isdir("{0}{1}".format(self.tempoutdir, filename_noext)):
            os.makedirs("{0}{1}".format(self.tempoutdir, filename_noext))
            os.makedirs("{0}{1}/images".format(self.tempoutdir, filename_noext))
            os.makedirs("{0}{1}/catalogues".format(self.tempoutdir, filename_noext))
            os.makedirs("{0}{1}/parameters".format(self.tempoutdir, filename_noext))

        self.tmp_dir = "{0}{1}/".format(self.tempoutdir, filename_noext, self.ccd)

        # if not os.path.isdir("{0}{1}".format(self.outdir, filename)):
        #     os.mkdir("{0}{1}".format(self.outdir, filename))
        #     os.mkdir("{0}{1}/images".format(self.outdir, filename))
        #     os.mkdir("{0}{1}/catalogues".format(self.outdir, filename))

        #conf_basefn = ntpath.basename(self.confmap)
        conf_basefn = self.confmap.split('/')[-1]
        imgfn = "{0}{1}/images/{2}".format(self.tempoutdir, filename_noext, filename)

        if not web:
            # Copy image file
            cmd = "cp {0} {1}".format(self.filename, imgfn)
            subprocess.call(cmd, shell=True)

            # Copy confidence map
            conffn = "{0}{1}/images/{2}".format(self.tempoutdir,
                                    filename_noext,
                                    conf_basefn)
            cmd = "cp {0} {1}".format(self.conffn, conffn)
            subprocess.call(cmd, shell=True)
        else:
            # Download image file
            imgfile = urllib.URLopener()
            logging.info("Downloading image file from {0}".format(self.filename))
            imgfile.retrieve(self.filename, "{0}{1}/images/{2}".format(self.tempoutdir, filename_noext, filename))
            imgfile.close()

            # Download confidence map
            conffn = "{0}{1}/images/conf.fit.fz".format(self.tempoutdir,
                                                filename_noext)#,
                                                #conf_basefn)
            conffile = urllib.URLopener()
            url = "http://www.iphas.org/data/images/confmaps/{0}.fz".format(self.confmap)
            logging.info("Downloading confidence map from {0}".format(url))
            conffile.retrieve(url, conffn)
            conffile.close()

        # funpack
        # cmd = "cp {0} {1}{2}/images/{2}_img.fits".format(self.imgfn,
        #                                                  self.tempoutdir,
        #                                                  self.filename)
        # subprocess.call(cmd, shell=True)
        # Delete any pre-existing decompressed versions of the image
        imgfn_new = imgfn.rstrip('.fz')
        conffn_temp = conffn.rstrip('.fz')
        confsplit1 = conffn.rstrip('.fz').split('/')
        confsplit2 = confsplit1[-1].split('.')
        conffn_new = '/'.join(confsplit1[:-1])+'/'+"-{0}.".format(self.ccd).join(confsplit2)
        try:
            os.remove(imgfn_new)
            logging.warning("Deleting pre-existing image file...")
        except OSError:
            logging.info("No pre-existing image file, continuing...")
        try:
            os.remove(conffn_new)
            logging.warning("Deleting pre-existing confidence file...")
        except OSError:
            logging.info("No pre-existing confidence file, continuing...")

        cmd = "funpack -D {0}".format(imgfn)
        subprocess.call(cmd, shell=True)
        cmd = "funpack -D {0}".format(conffn)
        subprocess.call(cmd, shell=True)

        conf = fits.open(conffn_temp)
        for ext in [i for i in [1,2,3,4] if i != self.ccd][::-1]:
            conf.pop(ext)

        conf.writeto("{0}".format(conffn_new), clobber=True)
        conf.close()
        os.remove(conffn_temp)

        return conffn_new, imgfn_new

    def imcore(self, imgfn, raw=True, M=None, magstart=None, delete=False):
        outdir = self.tempoutdir
        if raw:
            ofn = "{0}catalogues/r{1}_ccd{2}_cat.fits".format(self.tmp_dir, self.runid, self.ccd)
        else:
            ofn = "{0}catalogues/r{1}_ccd{2}_{3}_{4:03}_cat.fits".format(self.tmp_dir, self.runid, self.ccd, magstart, M)

        cmd = "imcore {0} {1} {2} 4 1.25 --rcore=3.5 --filtfwhm=2 --cattype=1 --noell".format(imgfn, self.conffn, ofn)

        subprocess.call(cmd, shell=True)

        if delete:
            os.remove(imgfn)
        return ofn

    def stilts(self, fn):
        """Cuts down number of columns in imcore output. Necessary since
        cattype=1 (INT) returns table with three identically named cols."""
        basefn = fn.split('.f')[0]
        stiltsjar = pkg_resources.resource_filename(__name__, 'data/stilts.jar')
        cmd = "java -jar {0} tpipe in={1}.fits out={1}_reduced.fits cmd='addcol Mag Blank32; keepcols \"X_coordinate Y_coordinate Ellipticity Position_angle Peak_height Core2_flux Classification Statistic Skylev Skyrms Mag\"; select X_coordinate>10; select X_coordinate<2038; select Y_coordinate>10; select Y_coordinate<4086'".format(stiltsjar, basefn)
        subprocess.call(cmd, shell=True)
        return "{0}_reduced.fits".format(basefn)

    def combine(self, imcorefn, stiltsfn):
        """Combines the output from imcore and stilts"""
        imcore = fits.open(imcorefn)
        stilts = fits.open(stiltsfn)
        for i in reversed(range(41,104)):
            imcore[1].header.pop(i)
        TTYPE  = ["X_coordinate", "Y_coordinate", "Ellipticity",
                  "Position_angle", "Peak_height", "Core2_flux", "Classification",
                  "Statistic", "Skylev", "Skyrms", "Mag"]
        TUNIT = ["Pixels", "Pixels", " ", "Degrees", "Counts", "Counts",
                 "Flag", "N-sigma", "Counts", "Counts", "Mag"]
        for i in range(11):
            imcore[1].header[8 + 3 * i] = TTYPE[i]
            imcore[1].header[9 + 3 * i] = "1E"
            imcore[1].header[10 + 3 * i] = TUNIT[i]
        imcore[1].header['NAXIS2'] = len(stilts[1].data)
        imcore[1].header['TFIELDS'] = 11
        imcore[1].header['NAXIS1'] = 44

        stilts[1].header = imcore[1].header
        stilts[0].header = imcore[0].header
        warnings.resetwarnings()
        warnings.filterwarnings('ignore', category=UserWarning, append=True)
        stilts.writeto(imcorefn, clobber=True)
        warnings.resetwarnings()
        warnings.filterwarnings('always', category=UserWarning, append=True)
        os.remove(stiltsfn)

    def addmags(self, catfn, column='Mag'):
        f = fits.open(catfn)
        if 'MAGS' in (f[0].header).keys():
            return
        header = f[1].header
        data = f[1].data
        zp = header['MAGZPT'] - (header['AIRMASS'] - 1) * header['EXTINCT']
        exp = header['EXPTIME']
        cor = header['APCOR2'] + header['PERCORR']
        for obj in range(len(f[1].data)):
            data[obj][column] =  zp - 2.5*np.log10(data[obj]['Core2_flux']/ exp) + cor
        f[1].data = data
        f[0].header['MAGS'] = "T"
        f.writeto(catfn, clobber=True)
        return

    def line(self, x, m, c):
        return m*x + c

    def magpeak(self):
        """docstring for magpeak"""
        mask = np.where((self.mag > 14) & (self.mag < 19))
        popt, pcov = curve_fit(self.line, self.mag[mask], np.log10(self.peak[mask]))
        self.parameters['popt_magpeak'] = popt

    def magflux(self):
        """docstring for magpeak"""
        mask = np.where((self.mag > 14) & (self.mag < 19))
        popt, pcov = curve_fit(self.line, self.mag[mask], np.log10(self.flux[mask]))
        self.parameters['popt_magflux'] = popt

    def posang(self):
        def gaussian(x, a, b, c):
            return a*np.exp(-0.5*(x-b)**2/(c**2))

        posang = self.cat[1].data['Position_angle']
        try:
            pa_bins = np.arange(0, 180, 10)
            pa_hst, pa_bine = np.histogram(posang, pa_bins)
            pa_binc = (pa_bine[1:] + pa_bine[:-1]) / 2
            popt, pcov = curve_fit(gaussian, pa_binc, pa_hst)
        except RuntimeError:
            popt = [0, 0, 0]
        if popt[1] == 0 or popt[1] < 0 or popt[1] > 180:
            popt[1] = 0.1
        self.parameters['popt_posang'] = popt

    def gethead(self):
        """docstring for gethead"""

        def fwhms(fwhm, ell):
            f1 = (2 * fwhm) / (2 - ell)
            f2 = f1 * (1 - ell)
            return f1, f2

        def fwhm2sigma(fwhm):
            return fwhm / (2 * np.sqrt(2 * np.log(2)))

        self.parameters['fwhm'] = self.cat[1].header['SEEING']
        self.parameters['ell'] = self.cat[1].header['ELLIPTIC']
        fwhm_1, fwhm_2 = fwhms(self.parameters['fwhm'], self.parameters['ell'])
        self.parameters['sig_1'], self.parameters['sig_2'] = fwhm2sigma(fwhm_1), fwhm2sigma(fwhm_2)

    def gaussian2d(self, height, x0, y0, sig_1, sig_2, theta):
        """Returns 2D elliptical gaussian which will be used to simulate a stellar source.
        Height    : Peak height of object's PSF
        x0, y0    : Central peak position in x and y
        sig_1/2   : Sigmas of gaussian along semi-major and semi-minor axes
        theta     : Position angle. SUPPLY IN DEGREES!
        """
        theta = np.radians(theta)
        A = (np.cos(theta) / sig_1)**2. + (np.sin(theta) / sig_2)**2.
        B = (np.sin(theta) / sig_1)**2. + (np.cos(theta) / sig_2)**2.
        C = (2.0 * np.sin(theta) * np.cos(theta) * (1. / (sig_1**2.) - 1. / (sig_2**2.)))
        return lambda x, y: (height * np.exp(-0.5 * (A * ((x - x0)**2) + B * ((y - y0)**2) + C * (x - x0) * (y - y0))))

    def genpeak(self, mag):
        h = self.line(mag, self.parameters['popt_magpeak'][0], self.parameters['popt_magpeak'][1])
        flux = np.power(10, h)
        return flux

    def genflux_corr(self, mag, flux):
        flux = np.sum(flux)
        h = self.line(mag, self.parameters['popt_magflux'][0], self.parameters['popt_magflux'][1])
        flux_fit = np.power(10, h)
        corr = flux_fit/flux
        return corr

    def genXY(self):
        """
        Generate random set of X, Y pixel coordinates. Using empirically determined zones-of-avoidance,
        ensure that the generated coordinates fall into the usable region of the WFC CCD under consideration.
        """
        X, Y = self.randomXY()
        if self.ccd == 1:
            if X > (1.6667 * Y) + 1558:
                X, Y = self.genXY()
        elif self.ccd == 2:
            if X > (0.6667 * Y) + 3664:
                X, Y = self.genXY()
        elif self.ccd == 3:
            if (X < (-1.1617 * Y) + 1882) or (X < (0.66667 * Y) - 2190) or (Y < 108):
                X, Y = self.genXY()
        return X, Y

    def randomXY(self):
        X = np.random.uniform(11, self.dim[1] - 11)
        Y = np.random.uniform(11, self.dim[0] - 11)
        return X, Y

    def generateobj(self, peak):
        theta = np.random.normal(self.parameters['popt_posang'][0], self.parameters['popt_posang'][1])
        return self.gaussian2d(peak, 10, 10, self.parameters['sig_1'], self.parameters['sig_2'], -theta)(*np.indices((21, 21)))

    def runsynthesis(self):
        xs, ys, mags, ms, magstarts = [], [], [], [], []
        for i in range(len(self.magbins) - 1):
            b = [self.magbins[i], self.magbins[i+1]]
            N = self.N[i]
            M = self.M[i]
            x, y, mag, m_b, magstart = self.synthesise(n=[N, M], mag_bin=b)
            xs += x
            ys += y
            mags += mag
            ms += m_b
            magstarts += magstart
        result = Table()
        result['x'] = xs
        result['y'] = ys
        result['mag'] = mags
        result['m'] = ms
        result['magstart'] = magstarts
        result.write("{0}parameters/parameters-r{1}-{2}.fits".format(self.tmp_dir, self.runid, self.ccd), overwrite=True)

    def synthesise(self, n=[100,10], mag_bin=None):

        # For checking Core2_flux, need pixel mask.
        x, y = np.indices((21, 21)) - 10
        dist = np.sqrt((x**2)+(y**2))
        mask = dist < 3.5   # Core radius is 3.5 pix

        mag_start = mag_bin[0]
        imgdir = "{0}images/".format(self.tmp_dir)
        x = []
        y = []
        mags = []
        peaks = []
        ms = []
        magstarts = []
        for i in range(n[1]):
            img_temp = fits.open("{0}".format(self.imgfn))
            for j in range(n[0]):
                logging.debug("Run {0:02}/{1:02}, {2:03}/{3:03}".format(i+1, n[1], j+1, n[0]))
                # Generate parameters for artificial object
                mag = np.random.uniform(mag_bin[0], mag_bin[1])
                X, Y = self.genXY()
                peak = self.genpeak(mag)
                # Create synthetic object "image"
                data = self.generateobj(peak)
                corr = self.genflux_corr(mag, data[mask])
                data = data*corr
                img_temp[1].data[Y - 10:Y + 11, X - 10:X + 11] += data
                ms.append(i)
                x.append(X)
                y.append(Y)
                mags.append(mag)
                magstarts.append(mag_start)
                peaks.append(peak)

            outfn = "{0}{1}_{2}_{3:03}.fits".format(imgdir, self.runid, mag_start, i)
            warnings.resetwarnings()
            warnings.filterwarnings('ignore', category=UserWarning, append=True)
            img_temp.writeto(outfn, clobber=True)
            warnings.resetwarnings()
            warnings.filterwarnings('always', category=UserWarning, append=True)
            # Keep a record of coordinates, magnitudes and peak fluxes
            # Create catalogue from new image
            imcorefn = self.imcore(outfn, False, i, magstart=mag_start, delete=True)
            stiltsfn = self.stilts(imcorefn)
            self.combine(imcorefn, stiltsfn)
            self.addmags(imcorefn)
        return x, y, mags, ms, magstarts

    def runrecovery(self):
        data = fits.getdata("{0}parameters/parameters-r{1}-{2}.fits".format(self.tmp_dir, self.runid, self.ccd))
        x, y, mag, m, magstart = data['x'], data['y'], data['mag'], data['m'], data['magstart']
        recovery, diffs = self.checkrecovery(x, y, mag, magstart, m)
        result = Table()
        result['x'] = x
        result['y'] = y
        result['mag'] = mag
        result['recovery'] = recovery
        result['diff'] = diffs
        #hdf5path = "result"
        recoveryfn = "{0}recovery-r{1}-{2}.fits".format(self.outdir, self.runid, self.ccd)
        result.write(recoveryfn, overwrite=True)
        #recoveryfn = "{0}recovery-r{1}-{2}.hdf5".format(self.outdir, self.runid, self.ccd)
        #result.write(recoveryfn, path=hdf5path, overwrite=True)
        return result

    def checkrecovery(self, x=None, y=None, mag=None, magstart=None, m=None):
        """Takes lists of x, y coordinates and corresponding magnitudes.
        Returns boolean array indicating which objects were detected."""

        recovery = np.ones((len(mag)), dtype=bool)
        diffs = np.ones((len(mag)), dtype='<f8') * 100  # Expected mag. - measured
        imcorefn = self.imcorefn[0]
        for obj in range(len(mag)):
            fn = "{0}catalogues/r{1}_ccd{2}_{3}_{4:03}_cat.fits".format(self.tmp_dir, self.runid, self.ccd, magstart[obj], m[obj])
            #fn = "{0}ccd{1}_{2}_{3:03}_cat.fits".format(imcorefn, self.ccd, magstart[obj], n[obj])
            f = fits.getdata(fn, 1)
            coord_mask = np.where((f['X_coordinate'] > x[obj] - self.dist) &
                                  (f['X_coordinate'] < x[obj] + self.dist) &
                                  (f['Y_coordinate'] > y[obj] - self.dist) &
                                  (f['Y_coordinate'] < y[obj] + self.dist))
            f = f[coord_mask]
            mag_mask = np.where((f['Mag'] > mag[obj] - self.magdist) &
                                (f['Mag'] < mag[obj] + self.magdist))
            f = f[mag_mask]
            if len(f) == 0:
                recovery[obj] = False
            else:
                differences = mag[obj] - f['Mag']
                diffs[obj] = np.min(differences)
        mask = np.where(diffs != 100.)
        # Since seeing causes a constant shift to magnitudes, shift by the median difference to get onto right scale.
        mdn = np.median(diffs[mask])
        diffs -= mdn
        return recovery, diffs

def process_run(run, ccd=4, **kwargs):
    """
    Simulate sources for an IPHAS CCD given its run number, returning tables containing the recovered objects.

    The returned tables will provide the simulated magnitudes, and the nearest source with the magnitude closest to the
    simulated magnitude. The user is then free to choose their own threshold as a cutoff to detemine if sources have
    been recovered successfully, thereby giving a measure of incompleteness for the magnitude bins simulated.

    :param run: Integer value for a given IPHAS run. These can be obtained from www.iphas.org.
    :param ccd: The CCD from the given run should be used to measure incompleteness. Default is CCD 4.
    :param kwargs: Optional arguments:
                   magbins: The start of magnitude bins for testing incompletness.
                            Default is 0.25 mag bins between 12.0 and 21.25 (for r and H-alpha), 11.0 and 20.25 (for i)
                   N      : Array of same length as magbins, specifying the number of sources to simulate in the IPHAS
                            image for each magnitude bin. TIP: Don't make this number so large that the resulting image
                            is so full of artificial sources that it no longer bears any resemblance to the original
                            IPHAS image. If you want to increase the number of simulated sources while keeping this
                            value low, consider increasing M instead (see below).
                   M      : Array of same length os magbins, specifying the number of times each magnitude bin should
                            be simulated (i.e. N[i] sources will be simulated M[i] times for magnitude bin i). Bear in
                            mind that the simulation script will need to call imcore M times, causing an increase in
                            run time.
                   dist   : The distance (in pixels) from the simulated source to expect recovered simulated sources.
                            Default is 3 px.
                   magdist: When determining if source has been recovered, magdist serves as an absolute threshold,
                            specifying what delta magnitude beyond which not to record nearby objects. Default is 2 mag.
                            e.g. if a source is detected within 'dist' of the artificial object, but the magnitude
                            difference between the artifical source and the recovered object is greater than 'magdist',
                            it will not be recorded in the results. This should probably not be used as the magnitude
                            delta you will eventually use to determine actual artifical source 'recovery' - this is just
                            designed to cut down on the number of results rows needed (by choosing a 'magdist' larger
                            than your expected final magnitude delta threshold, you will have leeway to change your
                            threshold later).
    :return:
    """


    iphas_images = fits.getdata(
        pkg_resources.resource_filename(__name__,
                                        'data/iphas-images.fits')
    )
    rows = iphas_images[np.where(iphas_images['run']==run)]
    if len(rows) == 0:
        logging.error("Can't find run {0} in iphas-images.fits".format(run))
        sys.exit(1)
    row = rows[np.where(rows['ccd'] == ccd)]
    if len(row) == 0:
        logging.error("Can't find CCD {0} for run {1} in iphas-images.fits".format(ccd, run))
        sys.exit(1)
    else:
        row = row[0]

    kwargs['web'] = True

    # Assign default arguments if they haven't been provided.
    if not kwargs.has_key('magbins'):
        band = row['band']
        if band in ['r', 'halpha']:
            kwargs['magbins'] = np.arange(12.0, 21.25, 0.25)
        elif band == 'i':
            kwargs['magbins'] = np.arange(11.0, 20.25, 0.25)
        else:
            logging.error(
                'This run\'s band is not a standard IPHAS filter. If you are sure this is the run you want to process, please supply your own \'magbin\' array.')
            sys.exit(1)
    if not kwargs.has_key('N'):
        kwargs['N'] = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                       20, 20, 20, 20, 20, 20, 30, 30, 30, 30, 30, 30,
                       40, 40, 40, 40, 50, 50, 50, 50, 50, 50, 50, 50, 50]
    if not kwargs.has_key('M'):
        kwargs['M'] = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                       10, 10, 10, 10, 10, 10, 15, 15, 20, 20, 20, 20, 20]
    if not kwargs.has_key('dist'):
        kwargs['dist'] = 3
    if not kwargs.has_key('magdist'):
        kwargs['magdist'] = 2

    if  not len(kwargs['magbins']) == len(kwargs['M']) == len(kwargs['N']):
        logging.error('The arrays \'magbin\', \'N\' and \'M\' must be of the same length.')
        sys.exit(1)

    simulation = Field(row['url'], row['confmap'], run, **kwargs)
    simulation.runsynthesis()
    if kwargs.has_key('outdir'):
        simulation.outdir = kwargs['outdir']
    simulation.runrecovery()