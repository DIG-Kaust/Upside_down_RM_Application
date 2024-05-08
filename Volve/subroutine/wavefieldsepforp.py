import numpy as np
import matplotlib.pyplot as plt
from pylops.waveeqprocessing.wavedecomposition import _obliquity2D


def wavefield_sepforp(vz, p, dt, dr, rho, vel, nffts,
                         critical, ntaper, verb=False, plotflag=False):
    r"""Up/down wavefield separation

    Separate multi-component seismic data in their up- and down-going particle
    velocity components

    Parameters
    ----------
    p : :obj:`numpy.ndarray`
        Pressure data of size :math:`[n_s \times n_r \times n_t]`
    vz : :obj:`numpy.ndarray`
        Vertical particle velocity data of size
        :math:`[n_s \times n_r \times n_t]`
    dt : :obj:`float`
        Time sampling
    dr : :obj:`float`
        Receiver sampling
    rho : :obj:`float`
        Density along the receiver array (must be constant)
    vel : :obj:`float`
        Velocity along the receiver array (must be constant)
    nffts : :obj:`tuple`, optional
        Number of samples along the wavenumber and frequency axes
    critical : :obj:`float`, optional
        Percentage of angles to retain in obliquity factor. For example, if
        ``critical=100`` only angles below the critical angle
        :math:`|k_x| < \frac{f(k_x)}{vel}` will be retained
    ntaper : :obj:`float`, optional
        Number of samples of taper applied to obliquity factor around critical
        angle
    verb : :obj:`bool`, optional
        Verbosity
    plotflag : :obj:`bool`, optional
        Plotting flag, if ``True`` plot results for the middle shot

    Returns
    -------
    vzup : :obj:`numpy.ndarray`
        Upgoing particle velocity data of size
        :math:`[n_s \times n_r \times n_t]`
    vzdown : :obj:`numpy.ndarray`
        Downgoing particle velocity data of size
        :math:`[n_s \times n_r \times n_t]`

    """
    ns, nr, nt = vz.shape

    FFTop, OBLop = \
        _obliquity2D(nt, nr, dt, dr, 1/rho, 1/vel,
                     nffts=nffts, critical=critical,
                     ntaper=ntaper, composition=True)

    pup, pdown = np.zeros_like(p), np.zeros_like(p)
    for isrc in range(ns):
        if verb:
            print('Working with source %d' % isrc)
        # FK transform
        VZ = FFTop * vz[isrc].ravel()

        # Scale Vz
        VZ_obl = OBLop * VZ.ravel()
        vz_obl = FFTop.H * VZ_obl
        vz_obl = np.real(vz_obl.reshape(nr, nt))

        # Separation
        pdown[isrc] = (p[isrc] + vz_obl) / 2
        pup[isrc] = (p[isrc] - vz_obl) / 2

        if plotflag and isrc == ns // 2:
            fig, axs = plt.subplots(1, 2, figsize=(9, 6))
            axs[0].imshow(vz_obl.T, cmap='gray',
                          vmin=-0.1 * np.abs(vz_obl).max(),
                          vmax=0.1 * np.abs(vz_obl).max(),
                          extent=(0, nr, 0, nt))
            axs[0].set_title(r'$vz$')
            axs[0].axis('tight')
            axs[1].imshow(p[isrc].T, cmap='gray',
                          vmin=-0.1 * np.abs(p).max(),
                          vmax=0.1 * np.abs(p).max(),
                          extent=(0, nr, 0, nt))
            axs[1].set_title(r'$pobl$')
            axs[1].axis('tight')

            fig, axs = plt.subplots(1, 2, figsize=(9, 6))
            axs[0].imshow(pup[isrc].T, cmap='gray',
                          vmin=-0.1 * np.abs(p).max(),
                          vmax=0.1 * np.abs(p).max(),
                          extent=(0, nr, 0, nt))
            axs[0].set_title(r'$pup$')
            axs[0].axis('tight')
            axs[1].imshow(pdown[isrc].T, cmap='gray',
                          vmin=-0.1 * np.abs(p).max(),
                          vmax=0.1 * np.abs(p).max(),
                          extent=(0, nr, 0, nt))
            axs[1].set_title(r'$pdown$')
            axs[1].axis('tight')

            plt.figure(figsize=(14, 3))
            plt.plot(vz_obl[nr // 2], 'r', lw=2, label=r'$vz$')
            plt.plot(p[isrc, nr // 2], '--b', lw=2, label=r'$p_z$')
            plt.xlim(0, nt // 4)
            plt.legend()
            plt.figure(figsize=(14, 3))
            plt.plot(pup[isrc, nr // 2], 'r', lw=2, label=r'$p^-$')
            plt.plot(pdown[isrc, nr // 2], 'b', lw=2, label=r'$p^+$')
            plt.xlim(0, nt // 4)
            plt.legend()

    return pup, pdown
