import numpy as np
import os
import time
import multiprocessing as mp

from multiprocessing import set_start_method
from multiprocessing import get_context
from scipy.signal import convolve, filtfilt
from pylops.waveeqprocessing.marchenko import directwave
from pylops.waveeqprocessing.mdd import MDD
from subroutine.marchenko import Marchenko
from subroutine.raymarchenko import RayleighMarchenko
from subroutine.anglegather import AngleGather
from subroutine.raymarchenko_upd_sparse_l2 import RayleighMarchenko_upd


def _imaging_depth_level(vsx, vsz, r, s, dr, ds, dt, nt, iava, nrava, vel,
                         toff, nsmooth, wav, wav_c, niter, nfmax,
                         igaths, nalpha,
                         data, kind='rmck_upd', jt=1, ivsz=None, verb=False):
                         
    print('ivsz = ',ivsz)
    
    if ivsz is not None and verb==True:
        print('Working on depth level %d' % ivsz)

    nr, ns = r.shape[1], s.shape[1]
    nvsx = len(vsx)
    dvsx = vsx[1] - vsx[0]
    ngath = len(igaths)

    if kind in ['mck', 'nmck']:
        directVS = np.sqrt((vsx - r[0][:, np.newaxis]) ** 2 +
                           (vsz - r[1][:, np.newaxis]) ** 2) / vel

        G0 = np.zeros((nr, nvsx, nt))
        for ivs in range(nvsx):
            G0[:, ivs] = directwave(wav, directVS[:, ivs], nt, dt,
                                    nfft=int(2 ** (np.ceil(np.log2(nt))))).T

        mck = Marchenko(data['R'], dt=dt, dr=dr, nfmax=nfmax, wav=wav,
                        toff=toff, nsmooth=nsmooth,
                        saveRt=False, prescaled=False)
        f1_inv_minus, f1_inv_plus, p0_minus, g_inv_minus, g_inv_plus = \
            mck.apply_multiplepoints(directVS, nfft=2 ** 11, rtm=True,
                                     greens=True, dottest=False,
                                     **dict(iter_lim=niter,
                                            show=False))

    elif kind in ['rmck', ]:
        directVSr = np.sqrt((vsx - r[0][:, np.newaxis]) ** 2 +
                            (vsz - r[1][:, np.newaxis]) ** 2) / vel
        directVSs = np.sqrt((vsx - s[0][:, np.newaxis]) ** 2 +
                            (vsz - s[1][:, np.newaxis]) ** 2) / vel

        G0rec = np.zeros((nr, nvsx, nt))
        for ivs in range(nvsx):
            G0rec[:, ivs] = directwave(wav, directVSr[:, ivs], nt, dt,
                                       nfft=int(2 ** (np.ceil(np.log2(nt))))).T

        G0 = np.zeros((ns, nvsx, nt))
        for ivs in range(nvsx):
            G0[:, ivs] = directwave(wav, directVSs[:, ivs], nt, dt,
                                    nfft=int(2 ** (np.ceil(np.log2(nt))))).T

        rm = RayleighMarchenko(data['Vzd'], data['Vzu'], dt=dt, dr=dr,
                               nfmax=nfmax, wav=wav, toff=toff,
                               nsmooth=nsmooth, saveVt=False, prescaled=False)
        print('niter is:',niter)
        f1_inv_minus, f1_inv_plus, p0_minus, g_inv_minus, g_inv_plus = \
            rm.apply_multiplepoints(directVSs, directVSr, G0=G0rec,
                                    rtm=True, greens=True,
                                    dottest=False,
                                    **dict(iter_lim=niter,
                                           show=False))
    
    elif kind in ['rmck_upd', ]:
        print('rmck_upd start')
        directVSr = np.sqrt((vsx - r[0][:, np.newaxis]) ** 2 +
                            (vsz - r[1][:, np.newaxis]) ** 2) / vel
        directVSs = np.sqrt((vsx - s[0][:, np.newaxis]) ** 2 +
                            (vsz - s[1][:, np.newaxis]) ** 2) / vel
        FirstVSmr = np.sqrt((vsx - r[0][:, np.newaxis]) ** 2 +
                            (vsz + r[1][:, np.newaxis]) ** 2) / vel

        G0 = np.zeros((nr, nvsx, nt))
        for ivs in range(nvsx):
            G0[:, ivs] = directwave(wav, FirstVSmr[:, ivs], nt, dt,
                                       nfft=int(2 ** (np.ceil(np.log2(nt))))).T
        G0 = G0[iava]
        
        G0src = np.zeros((ns, nvsx, nt))
        for ivs in range(nvsx):
            G0src[:, ivs] = directwave(wav, directVSs[:, ivs], nt, dt,
                                    nfft=int(2 ** (np.ceil(np.log2(nt))))).T

        rm = RayleighMarchenko_upd(data['Vzd'], data['Vzu'], dt=dt, ds=ds, iava=iava, nrava=nrava,
                                  nfmax=nfmax, wav=wav, toff=toff,
                                  nsmooth=nsmooth, saveVt=False, prescaled=False)

        f1_inv_minus, f1_inv_plus, p0_minus, g_inv_minus, g_inv_plus = \
            rm.apply_multiplepoints_upd(directVSs, FirstVSmr, G0=G0src, 
                                       rtm=True, greens=True, dottest=False,
                                       **dict(iter_lim=niter,
                                              show=False))

    # MDD
    _, Rss = MDD(G0[:, :, ::jt], p0_minus[:, :, nt - 1:][:, :, ::jt],
                 dt=jt * dt, dr=dvsx, twosided=True, adjoint=True, psf=False,
                 wav=wav[wav_c - 60:wav_c + 60],
                 nfmax=nfmax, dottest=False,
                 **dict(iter_lim=0, show=0))

    Rmck = MDD(g_inv_plus[:, :, nt - 1:][:, :, ::jt],
               g_inv_minus[:, :, nt - 1:][:, :, ::jt],
               dt=jt * dt, dr=dvsx, twosided=True, adjoint=False, psf=False,
               wav=wav[wav_c - 60:wav_c + 60],
               nfmax=nfmax, dottest=False,
               **dict(iter_lim=10, show=0))

    # Images
    iss = np.diag(Rss[:, :, nt - 1])
    imck = np.diag(Rmck[:, :, nt - 1])

    # Angle gathers
    ass = np.zeros((ngath, nalpha))
    amck = np.zeros((ngath, nalpha))
    for i, igath in enumerate(igaths):
        ass[i], angle, Ra = AngleGather(Rss.transpose(2, 0, 1), nvsx, nalpha,
                                        dt * jt, dvsx, igath, vel,
                                        plotflag=False)
        amck[i], angle, Ra = AngleGather(Rmck.transpose(2, 0, 1), nvsx, nalpha,
                                         dt * jt, dvsx, igath, vel,
                                         plotflag=False)
    
    return iss, imck, ass, amck

def MarchenkoImaging_upd(vsx, vsz, r, s, dr, ds, dt, nt, iava, nrava, vel,
                     toff, nsmooth, wav, wav_c, nfmax, igaths, nalpha, jt,
                     data, kind='rmck_upd', niter=10, nproc=1, nthreads=1):
    """Marchenko imaging

    Perform one of Marchenko's redatuming techiniques and apply multi-dimensional
    deconvolution to the retrieved Green's functions to obtain the local
    reflection response. This routine can be run for multiple depth levels
    and both images and angle-gathers can be produced as outputs

    Parameters
    ----------
    
    Returns
    -------
    iss : :obj:`numpy.ndarray`
        Single-scattering image
    imck : :obj:`numpy.ndarray`
        Marchenko image
    ass : :obj:`numpy.ndarray`, optional
        Single-scattering angle gathers
    amck : :obj:`numpy.ndarray`, optional
        Marchenko angle gathers
    """
    # Set threads
    os.environ["OMP_NUM_THREADS"] = str(nthreads)
    os.environ["MKL_NUM_THREADS"] = str(nthreads)

    # Imaging loop (single processor)
    nvsx, nvsz = len(vsx), len(vsz)

    t0 = time.time()
    if nproc == 1:
        results = [_imaging_depth_level(vsx, vsz[ivsz], r, s, dr, ds, dt, nt, iava, nrava, vel,
                                        toff, nsmooth, wav, wav_c, niter, nfmax,
                                        igaths, nalpha, data,
                                        kind=kind, jt=jt, ivsz=ivsz)
                   for ivsz in range(nvsz)]
    else:
        print('nvsz = ',nvsz)
        
        pool = mp.Pool(processes=nproc)
        results = pool.starmap(_imaging_depth_level, [(vsx, vsz[ivsz], r, s, dr, ds, dt, nt, iava, 
                                     nrava, vel, toff, nsmooth, wav, wav_c, niter, nfmax,
                                     igaths, nalpha, data, kind, jt, ivsz) 
                                     for ivsz in range(nvsz)])
        
    print('Elapsed time (mins): ', (time.time() - t0) / 60.)

    iss = np.vstack([results[ivsz][0] for ivsz in range(nvsz)])
    imck = np.vstack([results[ivsz][1] for ivsz in range(nvsz)])
    ass = np.concatenate([results[ivsz][2][:, np.newaxis, :]
                          for ivsz in range(nvsz)], axis=1)
    amck = np.concatenate([results[ivsz][3][:, np.newaxis, :]
                           for ivsz in range(nvsz)], axis=1)
    
    return iss, imck, ass, amck
    
    
    
