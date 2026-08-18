# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

r"""Woltring generalized cross-validation spline (GCVSPL), ported to NumPy.

OpenSim's ``InverseDynamicsTool`` differentiates filtered coordinate signals
with a :class:`GCVSpline` (a quintic natural spline). Its numerical core is
Woltring's GCVSPL package (Craven & Wahba 1979), which SimTK reimplements as
``SimTK::SplineFitter``. This module is a faithful 1:1 NumPy port of
``opensim-core`` ``OpenSim/Common/gcvspl.c`` restricted to the interpolating
case (error variance ``0``) that OpenSim uses by default, so the fitted spline
and its derivatives reproduce OpenSim's coordinate velocities and accelerations.

The lower-case helpers (:func:`basis`, :func:`prep`, :func:`splc`,
:func:`splder`, ...) keep the original GCVSPL routine names to ease
cross-referencing the source; :func:`fit_gcvspline` and :func:`eval_gcvspline`
are the public entry points.
"""

import numpy as np
import warp as wp

zero, half, one, two = 0.0, 0.5, 1.0, 2.0


def basis(m, n, x, b, q):
    m2m1 = 2 * m - 1
    if m == 1:
        for i in range(0, n + 1):
            b[(i - 1) * m2m1 + m - 1] = one
        return one
    mm1 = m - 1
    mp1 = m + 1
    m2 = 2 * m
    for l in range(1, n + 1):
        for j in range(-mm1, m + 1):
            q[j + m - 1] = zero
        q[mm1 + m - 1] = one
        if (l != 1) and (l != n):
            q[mm1 + m - 1] = one / (x[l] - x[l - 2])
        arg = x[l - 1]
        for i in range(3, m2 + 1):
            ir = mp1 - i
            v = q[ir + m - 1]
            if l < i:
                for j in range(l + 1, i + 1):
                    u = v
                    v = q[ir + m]
                    q[ir + m - 1] = u + (x[j - 1] - arg) * v
                    ir += 1
            j1 = (l - i + 1) if (l - i + 1) > 1 else 1
            j2 = (l - 1) if (l - 1) < (n - i) else (n - i)
            if j1 <= j2:
                if i < m2:
                    for j in range(j1, j2 + 1):
                        y = x[i + j - 1]
                        u = v
                        v = q[ir + m]
                        q[ir + m - 1] = u + (v - u) * (y - arg) / (y - x[j - 1])
                        ir += 1
                else:
                    for j in range(j1, j2 + 1):
                        u = v
                        v = q[ir + m]
                        q[ir + m - 1] = (arg - x[j - 1]) * u + (x[i + j - 1] - arg) * v
                        ir += 1
            nmip1 = n - i + 1
            if nmip1 < l:
                for j in range(nmip1, l):
                    u = v
                    v = q[ir + m]
                    q[ir + m - 1] = (arg - x[j - 1]) * u + v
                    ir += 1
        for j in range(-mm1, mm1 + 1):
            b[(l - 1) * m2m1 + j + m - 1] = q[j + m - 1]
    for i in range(1, mm1 + 1):
        for k in range(i, mm1 + 1):
            b[(i - 1) * m2m1 - k + m - 1] = zero
            b[(n - i) * m2m1 + k + m - 1] = zero
    bl = 0.0
    for i in range(1, n + 1):
        for k in range(-mm1, mm1 + 1):
            bl += abs(b[(i - 1) * m2m1 + k + m - 1])
    return bl / n


def prep(m, n, x, w, we):
    m2 = 2 * m
    mp1 = m + 1
    m2m1 = m2 - 1
    m2p1 = m2 + 1
    nm = n - m
    f1 = -one
    if m != 1:
        for i in range(2, m + 1):
            f1 *= -i
        for i in range(mp1, m2m1 + 1):
            f1 *= i
    i1 = 1
    i2 = m
    jm = mp1
    for j in range(1, n + 1):
        inc = m2p1
        if j > nm:
            f1 = -f1
            f = f1
        else:
            if j < mp1:
                inc = 1
                f = f1
            else:
                f = f1 * (x[j + m - 1] - x[j - m - 1])
        if j > mp1:
            i1 += 1
        if i2 < n:
            i2 += 1
        jj = jm
        ff = f
        y = x[i1 - 1]
        i1p1 = i1 + 1
        for i in range(i1p1, i2 + 1):
            ff = ff / (y - x[i - 1])
        we[jj - 1] = ff
        jj += m2
        i2m1 = i2 - 1
        if i1p1 <= i2m1:
            for l in range(i1p1, i2m1 + 1):
                ff = f
                y = x[l - 1]
                for i in range(i1, l):
                    ff = ff / (y - x[i - 1])
                for i in range(l + 1, i2 + 1):
                    ff = ff / (y - x[i - 1])
                we[jj - 1] = ff
                jj += m2
        ff = f
        y = x[i2 - 1]
        for i in range(i1, i2m1 + 1):
            ff = ff / (y - x[i - 1])
        we[jj - 1] = ff
        jj += m2
        jm += inc
    kl = 1
    n2m = m2p1 * n + 1
    for i in range(1, m + 1):
        ku = kl + m - i
        for k in range(kl, ku + 1):
            we[k - 1] = zero
            we[n2m - k - 1] = zero
        kl += m2p1
    jj = 0
    el = 0.0
    for i in range(1, n + 1):
        wi = w[i - 1]
        for _j in range(1, m2p1 + 1):
            jj += 1
            we[jj - 1] /= wi
            el += abs(we[jj - 1])
    return el / n


def bandet(e, m, n):
    m2p1 = 2 * m + 1
    if m <= 0:
        return
    for i in range(1, n + 1):
        di = e[(i - 1) * m2p1 + m]
        mi = m if m < (i - 1) else (i - 1)
        if mi >= 1:
            for k in range(1, mi + 1):
                di -= e[(i - 1) * m2p1 - k + m] * e[(i - k - 1) * m2p1 + k + m]
            e[(i - 1) * m2p1 + m] = di
        lm = m if m < (n - i) else (n - i)
        if lm >= 1:
            for l in range(1, lm + 1):
                dl = e[(i + l - 1) * m2p1 - l + m]
                km = (m - l) if (m - l) < (i - 1) else (i - 1)
                if km >= 1:
                    du = e[(i - 1) * m2p1 + l + m]
                    for k in range(1, km + 1):
                        du -= e[(i - 1) * m2p1 - k + m] * e[(i - k - 1) * m2p1 + l + k + m]
                        dl -= e[(l + i - 1) * m2p1 - l - k + m] * e[(i - k - 1) * m2p1 + k + m]
                    e[(i - 1) * m2p1 + l + m] = du
                e[(i + l - 1) * m2p1 - l + m] = dl / di


def bansol(e, y, c, m, n):
    m2p1 = 2 * m + 1
    nm1 = n - 1
    if m == 0:
        for i in range(1, n + 1):
            c[i - 1] = y[i - 1] / e[(i - 1) * m2p1 + m]
    elif m == 1:
        c[0] = y[0]
        for i in range(2, n + 1):
            c[i - 1] = y[i - 1] - e[(i - 1) * m2p1 - 1 + m] * c[i - 2]
        c[n - 1] = c[n - 1] / e[(n - 1) * m2p1 + m]
        for i in range(nm1, 0, -1):
            c[i - 1] = (c[i - 1] - e[(i - 1) * m2p1 + 1 + m] * c[i]) / e[(i - 1) * m2p1 + m]
    else:
        c[0] = y[0]
        for i in range(2, n + 1):
            mi = m if m < (i - 1) else (i - 1)
            d = y[i - 1]
            for k in range(1, mi + 1):
                d -= e[(i - 1) * m2p1 - k + m] * c[i - k - 1]
            c[i - 1] = d
        c[n - 1] /= e[(n - 1) * m2p1 + m]
        for i in range(nm1, 0, -1):
            mi = m if m < (n - i) else (n - i)
            d = c[i - 1]
            for k in range(1, mi + 1):
                d -= e[(i - 1) * m2p1 + k + m] * c[i + k - 1]
            c[i - 1] = d / e[(i - 1) * m2p1 + m]


def trinv(b, e, m, n):
    m2p1 = 2 * m + 1
    e[(n - 1) * m2p1 + m] = one / e[(n - 1) * m2p1 + m]
    for i in range(n - 1, 0, -1):
        mi = m if m < (n - i) else (n - i)
        dd = one / e[(i - 1) * m2p1 + m]
        for k in range(1, mi + 1):
            e[(n - 1) * m2p1 + k + m] = e[(i - 1) * m2p1 + k + m] * dd
            e[m - k] = e[(k + i - 1) * m2p1 - k + m]
        dd += dd
        for j in range(mi, 0, -1):
            du = zero
            dl = zero
            for k in range(1, mi + 1):
                du -= e[(n - 1) * m2p1 + k + m] * e[(i + k - 1) * m2p1 + j - k + m]
                dl -= e[(0) * m2p1 - k + m] * e[(i + j - 1) * m2p1 + k - j + m]
            e[(i - 1) * m2p1 + j + m] = du
            e[(j + i - 1) * m2p1 - j + m] = dl
            dd -= e[(n - 1) * m2p1 + j + m] * dl + e[(0) * m2p1 - j + m] * du
        e[(i - 1) * m2p1 + m] = dd / 2
    dd = zero
    for i in range(1, n + 1):
        mn = -m if m < (i - 1) else (1 - i)
        mp = m if m < (n - i) else (n - i)
        for k in range(mn, mp + 1):
            dd += b[(i - 1) * m2p1 + k + m] * e[(k + i - 1) * m2p1 - k + m]
    for k in range(1, m + 1):
        e[(n - 1) * m2p1 + k + m] = zero
        e[m - k] = zero
    return dd


def splc(m, n, y, w, var, p, eps, c, stat, b, we, el, bwe):
    m2p1 = 2 * m + 1
    m2m1 = 2 * m - 1
    dp = p
    stat[3] = p / (one + p)
    pel = p * el
    if (pel * eps) > one:
        stat[3] = one
        dp = one / (eps * el)
    if pel < eps:
        dp = eps / el
        stat[3] = 0
    for i in range(1, n + 1):
        km = -m if m < (i - 1) else (1 - i)
        kp = m if m < (n - i) else (n - i)
        for k in range(km, kp + 1):
            if abs(k) == m:
                bwe[(i - 1) * m2p1 + k + m] = dp * we[(i - 1) * m2p1 + k + m]
            else:
                bwe[(i - 1) * m2p1 + k + m] = b[(i - 1) * m2m1 + k + m - 1] + dp * we[(i - 1) * m2p1 + k + m]
    bandet(bwe, m, n)
    bansol(bwe, y, c, m, n)
    stat[2] = trinv(we, bwe, m, n) * dp
    trn = stat[2] / n
    esn = zero
    for i in range(1, n + 1):
        dt = -y[i - 1]
        km = (1 - m) if (m - 1) < (i - 1) else (1 - i)
        kp = (m - 1) if (m - 1) < (n - i) else (n - i)
        for k in range(km, kp + 1):
            dt += b[(i - 1) * m2m1 + k + m - 1] * c[i + k - 1]
        esn += dt * dt * w[i - 1]
    esn /= n
    stat[5] = esn / trn
    stat[0] = stat[5] / trn
    stat[1] = esn
    if var < zero:
        stat[4] = stat[5] - esn
        splcr = stat[0]
    else:
        stat[4] = esn - var * (two * trn - one)
        splcr = stat[4]
    return splcr


def search(n, x, t, l):
    if t < x[0]:
        return 0
    if t >= x[n - 1]:
        return n
    l = l if l > 1 else 1
    if l >= n:
        l = n - 1
    if t >= x[l - 1]:
        if t < x[l]:
            return l
        else:
            l += 1
            if t < x[l]:
                return l
            il = l + 1
            iu = n
    else:
        l -= 1
        if t >= x[l - 1]:
            return l
        else:
            il = 1
            iu = l
    while True:
        l = (il + iu) // 2
        if (iu - il) <= 1:
            return l
        if t < x[l - 1]:
            iu = l
        else:
            il = l


def splder(ider, m, n, t, x, c, l, q):
    m2 = 2 * m
    k = m2 - ider
    if k < 1:
        return zero, l
    l = search(n, x, t, l)
    tt = t
    mp1 = m + 1
    npm = n + m
    m2m1 = m2 - 1
    k1 = k - 1
    nk = n - k
    lk = l - k
    lk1 = lk + 1
    jl = l + 1
    ju = l + m2
    ii = n - m2
    ml = -l
    for j in range(jl, ju + 1):
        if (j >= mp1) and (j <= npm):
            q[j + ml - 1] = c[j - m - 1]
        else:
            q[j + ml - 1] = zero
    if ider > 0:
        jl -= m2
        ml += m2
        for i in range(1, ider + 1):
            jl += 1
            ii += 1
            j1 = 1 if 1 > jl else jl
            j2 = l if l < ii else ii
            mi = m2 - i
            j = j2 + 1
            if j1 <= j2:
                for _jin in range(j1, j2 + 1):
                    j -= 1
                    jm = ml + j
                    q[jm - 1] = (q[jm - 1] - q[jm - 2]) / (x[j + mi - 1] - x[j - 1])
            if jl < 1:
                i1 = i + 1
                j = ml + 1
                if i1 <= ml:
                    for _jin in range(i1, ml + 1):
                        j -= 1
                        q[j - 1] = -q[j - 2]
        for j in range(1, k + 1):
            q[j - 1] = q[j + ider - 1]
    if k1 >= 1:
        for i in range(1, k1 + 1):
            nki = nk + i
            ir = k
            jj = l
            ki = k - i
            nki1 = nki + 1
            if l >= nki1:
                for _j in range(nki1, l + 1):
                    q[ir - 1] = q[ir - 2] + (tt - x[jj - 1]) * q[ir - 1]
                    jj -= 1
                    ir -= 1
            lk1i = lk1 + i
            j1 = 1 if 1 > lk1i else lk1i
            j2 = l if l < nki else nki
            if j1 <= j2:
                for _j in range(j1, j2 + 1):
                    xjki = x[jj + ki - 1]
                    z = q[ir - 1]
                    q[ir - 1] = z + (xjki - tt) * (q[ir - 2] - z) / (xjki - x[jj - 1])
                    ir -= 1
                    jj -= 1
            if lk1i <= 0:
                jj = ki
                lk1i1 = 1 - lk1i
                for _j in range(1, lk1i1 + 1):
                    q[ir - 1] = q[ir - 1] + (x[jj - 1] - tt) * q[ir - 2]
                    jj -= 1
                    ir -= 1
    z = q[k - 1]
    if ider > 0:
        for j in range(k, m2m1 + 1):
            z *= j
    return z, l


def fit_gcvspline(x, y, half_order=3):
    """Interpolating (var=0) natural B-spline fit; returns coefficients c."""
    m = half_order
    n = len(x)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.ones(n)
    m2p1 = 2 * m + 1
    m2m1 = 2 * m - 1
    b = np.zeros(n * m2m1)
    we = np.zeros(n * m2p1)
    bwe = np.zeros(n * m2p1)
    c = np.zeros(n)
    stat = np.zeros(6)
    qwork = np.zeros(n * m2p1 + 8)
    bl = basis(m, n, x, b, qwork)
    el = prep(m, n, x, w, we)
    el /= bl
    splc(m, n, y, w, 0.0, 0.0, 1e-15, c, stat, b, we, el, bwe)
    return c


def eval_gcvspline(x, c, t, order=0, half_order=3):
    """Evaluate spline (or its ider-th derivative) at scalar t."""
    m = half_order
    n = len(x)
    q = np.zeros(2 * m + 4)
    z, _ = splder(order, m, n, float(t), np.asarray(x, float), c, 1, q)
    return z


# --------------------------------------------------------------------------- #
# Warp device evaluation of a fitted spline (matches :func:`eval_gcvspline`).
#
# ``fit_gcvspline`` (Woltring banded solve) stays on the host; only the de Boor
# evaluation -- the per-sample work that scales with the number of output times
# and columns -- runs on-device. ``half_order`` is fixed to 3 (quintic), the
# order OpenSim's ``InverseDynamicsTool`` uses, so the per-thread scratch is a
# fixed-length vector.
# --------------------------------------------------------------------------- #
_GCV_M = wp.constant(3)
_vecq = wp.types.vector(length=10, dtype=wp.float64)


@wp.func
def gcv_search(n: int, x: wp.array[wp.float64], t: wp.float64, l: int) -> int:
    """Locate the knot interval containing ``t`` (port of :func:`search`)."""
    if t < x[0]:
        return 0
    if t >= x[n - 1]:
        return n
    if l <= 1:
        l = 1
    if l >= n:
        l = n - 1
    il = int(1)
    iu = n
    if t >= x[l - 1]:
        if t < x[l]:
            return l
        l += 1
        if t < x[l]:
            return l
        il = l + 1
        iu = n
    else:
        l -= 1
        if t >= x[l - 1]:
            return l
        il = 1
        iu = l
    while True:
        l = (il + iu) // 2
        if (iu - il) <= 1:
            return l
        if t < x[l - 1]:
            iu = l
        else:
            il = l
    return l


@wp.func
def gcv_splder(
    ider: int, n: int, t: wp.float64, x: wp.array[wp.float64], coeffs: wp.array2d[wp.float64], col: int, l: int
) -> wp.float64:
    """Evaluate the ``ider``-th derivative of the fitted spline at ``t`` (port of :func:`splder`)."""
    m = _GCV_M
    m2 = 2 * m
    k = m2 - ider
    if k < 1:
        return wp.float64(0.0)
    l = gcv_search(n, x, t, l)
    tt = t
    mp1 = m + 1
    npm = n + m
    m2m1 = m2 - 1
    k1 = k - 1
    nk = n - k
    lk = l - k
    lk1 = lk + 1
    jl = l + 1
    ju = l + m2
    ii = n - m2
    ml = -l
    q = _vecq()
    for j in range(jl, ju + 1):
        if (j >= mp1) and (j <= npm):
            q[j + ml - 1] = coeffs[col, j - m - 1]
        else:
            q[j + ml - 1] = wp.float64(0.0)
    if ider > 0:
        jl -= m2
        ml += m2
        for i in range(1, ider + 1):
            jl += 1
            ii += 1
            j1 = wp.max(1, jl)
            j2 = wp.min(l, ii)
            mi = m2 - i
            j = j2 + 1
            if j1 <= j2:
                for _jin in range(j1, j2 + 1):
                    j -= 1
                    jm = ml + j
                    q[jm - 1] = (q[jm - 1] - q[jm - 2]) / (x[j + mi - 1] - x[j - 1])
            if jl < 1:
                i1 = i + 1
                j = ml + 1
                if i1 <= ml:
                    for _jin in range(i1, ml + 1):
                        j -= 1
                        q[j - 1] = -q[j - 2]
        for j in range(1, k + 1):
            q[j - 1] = q[j + ider - 1]
    if k1 >= 1:
        for i in range(1, k1 + 1):
            nki = nk + i
            ir = k
            jj = l
            ki = k - i
            nki1 = nki + 1
            if l >= nki1:
                for _j in range(nki1, l + 1):
                    q[ir - 1] = q[ir - 2] + (tt - x[jj - 1]) * q[ir - 1]
                    jj -= 1
                    ir -= 1
            lk1i = lk1 + i
            j1 = wp.max(1, lk1i)
            j2 = wp.min(l, nki)
            if j1 <= j2:
                for _j in range(j1, j2 + 1):
                    xjki = x[jj + ki - 1]
                    z = q[ir - 1]
                    q[ir - 1] = z + (xjki - tt) * (q[ir - 2] - z) / (xjki - x[jj - 1])
                    ir -= 1
                    jj -= 1
            if lk1i <= 0:
                jj = ki
                lk1i1 = 1 - lk1i
                for _j in range(1, lk1i1 + 1):
                    q[ir - 1] = q[ir - 1] + (x[jj - 1] - tt) * q[ir - 2]
                    jj -= 1
                    ir -= 1
    z = q[k - 1]
    if ider > 0:
        for j in range(k, m2m1 + 1):
            z *= wp.float64(j)
    return z


@wp.kernel
def gcv_eval_kernel(
    x: wp.array[wp.float64],
    coeffs: wp.array2d[wp.float64],
    out_times: wp.array[wp.float64],
    n: int,
    out: wp.array3d[wp.float64],
):
    """Evaluate a fitted spline's value, first, and second derivative on a grid.

    ``out[r, c, o]`` holds the ``o``-th derivative (0, 1, 2) of column ``c`` at
    ``out_times[r]``.
    """
    r, c = wp.tid()
    t = out_times[r]
    for order in range(3):
        out[r, c, order] = gcv_splder(order, n, t, x, coeffs, c, 1)


def eval_gcvspline_batch(x, coeffs2d, out_times, device=None):
    """Evaluate fitted splines (value/1st/2nd derivative) over a grid on-device.

    Args:
        x: Shared knot vector, shape ``[n]``.
        coeffs2d: Per-column fitted coefficients, shape ``[num_columns, n]``.
        out_times: Times to evaluate at, shape ``[num_out]``.
        device: Warp device (defaults to CPU).

    Returns:
        Array of shape ``[num_out, num_columns, 3]`` with the value, first, and
        second derivative of each column at each output time.
    """
    dev = wp.get_device(device) if device is not None else wp.get_device("cpu")
    x = np.ascontiguousarray(x, np.float64)
    coeffs2d = np.ascontiguousarray(coeffs2d, np.float64)
    out_times = np.ascontiguousarray(out_times, np.float64)
    n = x.shape[0]
    num_out, ncol = out_times.shape[0], coeffs2d.shape[0]
    d_x = wp.array(x, dtype=wp.float64, device=dev)
    d_c = wp.array(coeffs2d, dtype=wp.float64, device=dev)
    d_t = wp.array(out_times, dtype=wp.float64, device=dev)
    d_out = wp.zeros((num_out, ncol, 3), dtype=wp.float64, device=dev)
    wp.launch(gcv_eval_kernel, dim=(num_out, ncol), inputs=[d_x, d_c, d_t, n, d_out], device=dev)
    return d_out.numpy()
