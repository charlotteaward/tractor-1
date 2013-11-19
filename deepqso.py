from astrometry.util.fits import *
from glob import glob

''' Stripe82 deep QSO plates (RA 36-42):

Select from DR7 CAS, Stripe82 context:

select
objid, ra, dec, fracdev_r as fracdev, type as objc_type, modelmag_r,
devrad_r as theta_dev, devraderr_r as theta_deverr, devab_r as ab_dev, devaberr_r as ab_deverr,
devphi_r as phi_dev_deg,
exprad_r as theta_exp, expraderr_r as theta_experr,
expab_r as ab_exp, expaberr_r as ab_experr, expphi_r as phi_exp_deg,
resolve_status = 256,
nchild, flags_r as flags, flags as objc_flags,
run, camcol, field, obj as id, rerun
into mydb.deepqso from PhotoPrimary
where ra between 36 and 42
and dec between -1.3 and 1.3
and (run = 106 or run = 206)

-> deepqso.fits


Ran custom unwise coadds:

python -u unwise-coadd.py --width 3300 --height 1350 -o data/deepqso --dec=0 --ra 36.5 1000 > 36.log 2>&1 &
python -u unwise-coadd.py --width 3300 --height 1350 -o data/deepqso --dec=0 --ra 37.5 1000 > 37.log 2>&1 &
python -u unwise-coadd.py --width 3300 --height 1350 -o data/deepqso --dec=0 --ra 38.5 1000 > 38.log 2>&1 &
python -u unwise-coadd.py --width 3300 --height 1350 -o data/deepqso --dec=0 --ra 39.5 1000 > 39.log 2>&1 &
python -u unwise-coadd.py --width 3300 --height 1350 -o data/deepqso --dec=0 --ra 40.5 1000 > 40.log 2>&1 &
python -u unwise-coadd.py --width 3300 --height 1350 -o data/deepqso --dec=0 --ra 41.5 1000 > 41.log 2>&1 &

And defined a custom -atlas.fits:

# ra dec coadd_id
36.5 0 custom-0365p000
37.5 0 custom-0375p000
38.5 0 custom-0385p000
39.5 0 custom-0395p000
40.5 0 custom-0405p000
41.5 0 custom-0415p000

text2fits.py deepqso.txt deepqso-atlas.fits

Then run this script to create photometry files, and run sequels.py as usual,



'''

T = fits_table('deepqso.fits')
dra = 0.51
for ra in range(36, 42):
    ra = ra + 0.5
    Ti = T[(T.ra > (ra-dra)) * (T.ra < (ra+dra))]
    print len(Ti), 'in RA block', ra
    # We only use the flux at the bright limit, so Luptitudes vs Pogson doesn't matter
    Ti.modelflux = np.zeros((len(Ti),5), np.float32)
    Ti.modelflux[:,2] = 10.**((Ti.modelmag_r - 22.5)/-2.5)

    for c in ['fracdev', 'theta_dev', 'theta_deverr', 'theta_exp',  'theta_experr',
              'ab_dev', 'ab_deverr', 'ab_exp', 'ab_experr',
              'phi_dev_deg', 'phi_exp_deg']:
        x = Ti.get(c)
        y = np.zeros((len(Ti), 5), np.float32)
        y[:,2] = x
        Ti.set(c, y)

    Ti.writeto('deepqso-phot-temp/photoobjs-custom-0%i5p000.fits' % ra)
    

