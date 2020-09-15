import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt

def E(z, Om, w):
    r"""The usual :math:`E` factor that appears in the comoving distance integrand.

    Here we specialize to flat cosmologies with dark energy and matter, so
    :math:`E` is defined by

    .. math::

      E \equiv \sqrt{\Omega_m \left( 1 + z \right)^3 + \left( 1 - \Omega_m \right) \left( 1 + z \right)^{3(1+w)}}

    :param z: The redshift(s) at which to evaluate :math:`E`.

    :param Om: The dimensionless matter density (relative to the critical density).

    :param w: The dark energy equation of state parameter.
    """
    z = tt.as_tensor_variable(z)
    Om = tt.as_tensor_variable(Om)
    w = tt.as_tensor_variable(w)

    opz = 1.0 + z

    return tt.sqrt(Om*opz*opz*opz + (1-Om)*opz**(3*(1+w)))

def tt_trapz(ys, xs):
    r"""Theano implementation of :func:`np.trapz`.

    """
    ys = tt.as_tensor_variable(ys)
    xs = tt.as_tensor_variable(xs)

    dxs = tt.extra_ops.diff(xs)
    return tt.sum(0.5*dxs*(ys[1:]+ys[:-1]))

def tt_cumtrapz(ys, xs):
    r"""Theano implementation of :func:`scipy.integration.cumtrapz`.

    Compared to the `scipy` function, we fix `initial=0`, so the returned array
    has the *same* dimensions as `ys` and always contains `0` as its first
    element.

    """
    ys = tt.as_tensor_variable(ys)
    xs = tt.as_tensor_variable(xs)

    dxs = tt.extra_ops.diff(xs)
    result = tt.zeros_like(ys)
    result = tt.set_subtensor(result[1:], tt.cumsum(0.5*dxs*(ys[1:] + ys[:-1])))
    return result

def dCs(zs, dH, Om, w):
    r"""Compute the comoving distances at the given redshifts in a cosmology.

    The array of `zs` provided must have spacing sufficient to obtain an
    approximation to the integral in the comoving distance definition via the
    trapezoid rule.

    :param zs: Redshifts.  See note above about spacing.

    :param dH: The Hubble distance, :math:`d_H = c/H_0`.

    :param Om: Dimensionless matter density.

    :param w: Dark enery EOS parameter.

    :return: Array of comoving distances at the given redshifts (units
      correspond to `dH`).
    """
    zs = tt.as_tensor_variable(zs)
    dH = tt.as_tensor_variable(dH)
    Om = tt.as_tensor_variable(Om)
    w = tt.as_tensor_variable(w)

    return dH*tt_cumtrapz(1.0/E(zs, Om, w), zs)

def dLs(zs, dCs):
    """Convenience function to convert redshifts and comoving distances to
    luminosity distances.
    """
    return (1+zs)*dCs

def dVdz(zs, dCs, dH, Om, w):
    r"""Returns differential comoving volumes at the indicated redshifts/comoving
    distances.

    :param zs: Redshifts.

    :param dCs: Comoving distances.

    :param dH: Hubble distance, :math:`d_H \equiv c/H_0`.

    :param Om: Dimensionless matter density.

    :param w: Dark energy equation of state parameter.
    """
    return 4.0*np.pi*dCs*dCs*dCs*dH/E(zs, Om, w)

def tt_interp(x, xs, ys):
    x = tt.as_tensor_variable(x)
    xs = tt.as_tensor_variable(xs)
    ys = tt.as_tensor_variable(ys)

    i = tt.extra_ops.searchsorted(xs, x)

    # Fix up the ends
    i = tt.where(tt.eq(i, 0), 1, i)
    i = tt.where(tt.eq(i, xs.shape[0]), xs.shape[0]-1, i)

    xl = xs[i-1]
    xr = xs[i]
    yl = ys[i-1]
    yr = ys[i]

    r = (x - xl)/(xr - xl)
    return yl*(1-r) + yr*r

def p_z(zs, dCs, dH, Om, w):
    return (1+zs)**2.7/(1+((1+zs)/(1+1.9))**5.6)*dVdz(zs, dCs, dH, Om, w)/(1+zs)

def beta(z_horiz, zs, pzs):
    czs = tt_cumtrapz(pzs, zs)
    return tt_interp(z_horiz, zs, czs)

def make_model(dl, p_dl, zc, dl_horizon, H0_prior=None, Omh2_prior=None, fix_w=True):
    zmax = 10
    z = np.expm1(np.linspace(np.log(1.0), np.log(1.0+zmax), 1024))

    with pm.Model() as model:
        if H0_prior is None:
            H0 = pm.Uniform('H0', lower=35, upper=140, testval=70.0)
        else:
            H0 = pm.Interpolated('H0', H0_prior[0], H0_prior[1], testval=70.0)

        if Omh2_prior is None:
            Om = pm.Uniform('Om', lower=0, upper=1, testval=0.3)
            Omh2 = pm.Deterministic('Omh2', Om*(H0/100.0)*(H0/100.0))
        else:
            Omh2 = pm.Normal('Omh2', mu=Omh2_prior[0], sd=Omh2_prior[1], testval=Omh2_prior[0])
            Om = pm.Deterministic('Om', Omh2/((H0/100.0)*(H0/100.0)))

        if fix_w:
            w = -1.0
        else:
            w = pm.Uniform('w', lower=-2, upper=0, testval=-1.0)

        dH = 2.99792e5/H0 # Mpc

        dC = dCs(z, dH, Om, w)
        dL = dLs(z, dC)
        pz = p_z(z, dC, dH, Om, w)
        z_horizon = tt_interp(dl_horizon, dL, z)
        p = tt_interp(zc, z, pz)
        b = beta(z_horizon, z, pz)
        d = tt_interp(zc, z, dL)

        logl = pm.Interpolated.dist(dl, p_dl)

        pm.Potential('logl', logl.logp(d) + tt.log(p/b))

    return model
