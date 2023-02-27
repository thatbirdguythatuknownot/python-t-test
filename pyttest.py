"""
(Student's T-Test) Statistics module in pure Python.

-- CONSTANTS --

EPS = sys.float_info.epsilon
DBL_MAX = sys.float_info.max
INF = float('inf')
NAN = float('nan')
MACHEP = 2**-53
MAXLOG = math.log(2**1024)
MINLOG = math.log(2**-1022)

---------------

-- FUNCTIONS --

mean(arr)                    - sample mean (probably also population mean)
pdf(t, v)                    - probability density function
lnbeta(a, b)                 - natural logarithm (function) of beta function
labeta(a, b)                 - less accurate beta function
                               (= `exp(lnbeta(a, b))`)
beta(a, b)                   - beta function
rbeta(x, p, q)               - regularized incomplete beta function
invrbeta(yy, aa, bb)         - inverse function of `rbeta()`
stdtri(confidence, df)       - inverse function of `cdf()`
tcritv(alpha, df, tn=2)      - critical t value function
                               ("tail number" can either be 1 or 2)
tpval(t, df, tp=2)           - t-test p-value function
                               ("test type" can be:
                                0 for left-tail/lower-tailed t-test,
                                1 for right-tail/upper-tailed t-test,
                                2 for two-tail t-test [not symmetric about 0],
                                3 for two-tail t-test [symmetric about 0])
cdf(t, v)                    - cumulative distribution function
sstd(arr)                    - sample standard deviation function
pstd(arr)                    - population standard deviation function
pvar(*arrs)                  - pooled variance
                               (pooled standard deviation) function

- t-test functions returning (t statistic, degrees of freedom)
otstat(arr, apmean)           - one-sample t-test
                                (provide assumed population mean)
ptstat(arr1, arr2)           - paired samples t-test
itstat(arr1, arr2, eva=True) - independent samples t-test /
                               unpaired samples t-test / two-sample t-test
                               (provide "equal variance assumed")

---------------

-- INTERACTIVE MODE --

Enter interactive mode by directly running the file.
Interactive mode can be quitted by sending a Ctrl+Z then Enter (Windows)
or Ctrl+D then Enter (Linux).

----------------------

"""

from collections import defaultdict
from functools import cache, wraps
from math import exp, gamma, lgamma, log, trunc
from operator import sub
from sys import float_info
from types import FunctionType

__all__ = ['EPS', 'DBL_MAX', 'INF', 'NAN', 'MACHEP', 'MAXLOG', 'MINLOG',
           'mean', 'pdf', 'lnbeta', 'labeta', 'beta', 'rbeta', 'invrbeta',
           'stdtri', 'tcritv', 'tpval', 'cdf', 'sstd', 'pstd', 'pvar',
           'otstat', 'ptstat', 'itstat']

EPS = float_info.epsilon
DBL_MAX = float_info.max
INF = float('inf')
NAN = float('nan')

def oi(n): # optionally [convert to] integer
    return n if n % 1 else trunc(n)

# `@oideco` -> one-argument number return
# `@oideco(x)` -> multiple-argument number return; does not depend on `x`
def oideco(f=None, *args, **kwargs):
    if not isinstance(f, FunctionType) or args or kwargs:
        def oideco_wrap(f):
            @wraps(f)
            def wrap(*args, **kwargs):
                return tuple(map(oi, f(*args, **kwargs)))
            return wrap
        wrap = oideco_wrap
    else:
        @wraps(f)
        def wrap(*args, **kwargs):
            return oi(f(*args, **kwargs))
    return wrap

def oideco_C(f=None, *args, **kwargs): # oideco() w/ functools.cache()
    if not isinstance(f, FunctionType) or args or kwargs:
        def oideco_C_wrap(f):
            @wraps(f)
            @cache
            def wrap(*args, **kwargs):
                return tuple(map(oi, f(*args, **kwargs)))
            return wrap
        wrap = oideco_C_wrap
    else:
        @wraps(f)
        @cache
        def wrap(*args, **kwargs):
            return oi(f(*args, **kwargs))
    return wrap

@oideco_C
def rf(q, n): # rising factorial / pochhammer function
    if n:
        a = q
        for i in range(1, n):
            a *= q + i
        return a
    return 1

_flt = [1, 1, 2, 6, 24, 120, 720, 5040, 40320,
        362880, 3628800, 39916800, 479001600,
        6227020800, 87178291200, 1307674368000,
        20922789888000, 355687428096000,
        6402373705728000, 121645100408832000]
_flt_append = _flt.append

_plt = defaultdict(list)

@oideco_C
def ohf(a, b, c, z): # ordinary hypergeometric function (2F1)
    plt_z = _plt[z]
    plt_z_append = plt_z.append
    s = n = 0
    f = 1
    p = -1
    while s != p:
        p = s
        try:
            z_tn = plt_z[n]
        except IndexError:
            plt_z_append(z_tn := z**n)
        try:
            f_n = _flt[n]
        except IndexError:
            _flt_append(f_n := _flt[n-1] * n)
        s += rf(a, n)*rf(b, n)/rf(c, n) * (z_tn / f_n)
        n += 1
    return s

@oideco
def mean(arr):
    return sum(arr) / len(arr)

@oideco_C
def _func_0(v):
    if v > 1:
        b_val = 1
        if v & 1:
            a_val = v - 1
            for i in range(3, v, 2):
                a_val *= i - 1
                b_val *= i
            dv = 3.141592653589793
        else:
            a_val = 1
            for i in range(3, v, 2):
                a_val *= i
                b_val *= i - 1
            dv = 2
        return a_val / dv / v**(1/2) / b_val
    if v == 1:
        return 0.3183098861837907
    raise ValueError("math domain error")

@oideco_C
def pdf(t, v): # probability density function
    return _func_0(v) * (1 + t**2/v)**(~v / 2)

# https://github.com/canerturkmen/betaincder/blob/master/betaincder/c/beta.c
##----------------------------##
@oideco_C
def lnbeta(a, b): # ln of beta
    return lgamma(a) + lgamma(b) - lgamma(a + b)

@oideco_C
def beta(a, b): # beta function
    return exp(lnbeta(a, b))

##----------------------------##

# https://malishoaib.wordpress.com/tag/complete-beta-function/
##----------------------------##
# redefine beta() function because previous definition was a little inaccurate
# assign old beta() to `labeta` meaning "less accurate beta"
labeta = beta
@oideco_C
def beta(a, b): # beta function
    return gamma(a)*gamma(b) / gamma(a + b)

@oideco_C
def cfbeta(a, b, x): # contfractbeta
    bm = az = am = 1
    qab = a + b
    qap = a + 1
    qam = a - 1
    bz = 1 - qab*x/qap
    em = 1
    while True:
        tem = em + em
        a_ptem = a + tem
        ap = az + (d := em*(b-em)*x / ((qam + tem)*a_ptem))*am
        bp = bz + d*bm
        bpp = bp + (d := -(a+em)*(qab+em)*x / ((qap + tem)*a_ptem)) * bz
        aold = az
        if 2**-53 * abs(az := (ap + d*az) / bpp) > abs(az - aold):
            return az
        bz = 1
        am = ap / bpp
        bm = bp / bpp
        em += 1

@oideco_C
def rbeta(x, a, b): # regularized beta function
    if x not in {0, 1}:
        _beta = gamma(a+b)/gamma(a)/gamma(b) * x**a * (1-x)**b
        if x < (a+1) / (a+b+2):
            return _beta*cfbeta(a, b, x) / a
        else:
            return 1 - _beta*cfbeta(b, a, 1 - x)/b
    return x

##----------------------------##

# https://github.com/scipy/scipy/blob/main/scipy/special/cephes/
##----------------------------##

# const.c
#----------------------------#
# the variables aren't actually used in this code,
# just the values as a constant
MACHEP = 1.11022302462515654042e-16     # 2**-53
MAXLOG = 709.782712893383996843         # log(2**1024)
MINLOG = -708.396418532264106224        # log(2**-1022)
#----------------------------#

# polevl.h
#----------------------------#
@oideco
def polevl(x, coef, N):
    ans = coef[i := 0]
    while N:
        ans = ans*x + coef[i := i + 1]
        N -= 1
    return ans

@oideco
def p1evl(x, coef, N):
    ans = x + coef[i := 0]
    while N := N - 1:
        ans = ans*x + coef[i := i + 1]
    return ans

#----------------------------#

# ndtri.c
#----------------------------#

# approximation for 0 <= |y - 0.5| <= 3/8
P0 = [-59.9633501014107895267, 98.0010754185999661536,
      -56.6762857469070293439, 13.9312609387279679503,
      -1.23916583867381258016]
Q0 = [1.95448858338141759834, 4.67627912898881538453,
      86.3602421390890590575, -225.462687854119370527,
      200.260212380060660359, -82.0372256168333339912,
      15.9056225126211695515, -1.18331621121330003142]

# Approximation for interval z = sqrt(-2 log y ) between 2 and 8
# i.e., y between exp(-2) = .135 and exp(-32) = 1.27e-14.
P1 = [4.05544892305962419923, 31.5251094599893866154,
      57.1628192246421288162, 44.0805073893200834700,
      14.6849561928858024014, 2.18663306850790267539E0,
      -0.140256079171354495875, -0.0350424626827848203418,
      -0.000857456785154685413611]
Q1 = [15.7799883256466749731, 45.3907635128879210584,
      41.3172038254672030440, 15.0425385692907503408,
      2.50464946208309415979, -0.142182922854787788574,
      -0.0380806407691578277194, -0.000933259480895457427372]

# Approximation for interval z = sqrt(-2 log y ) between 8 and 64
# i.e., y between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890.
P2 = [3.23774891776946035970, 6.91522889068984211695,
      3.93881025292474443415, 1.33303460815807542389,
      0.201485389549179081538, 0.0123716634817820021358,
      0.000301581553508235416007, 2.65806974686737550832e-6,
      6.23974539184983293730e-9]
Q2 = [6.02427039364742014255, 3.67983563856160859403,
      1.37702099489081330271, 0.216236993594496635890,
      0.0134204006088543189037, 0.000328014464682127739104,
      2.89247864745380683936e-6, 6.79019408009981274425e-9]

@oideco_C
def ndtri(y):
    if y == 0: return -INF
    if y == 1: return INF
    if y < 0 or y > 1: return NAN
    code = True
    if y > 1 - 0.13533528323661269189: # 1 - exp(-2)
        y = 1 - y
        code = False
    if y > 0.13533528323661269189:
        y -= 0.5
        return ((y + y*((y2 := y * y)*polevl(y2, P0, 4) / p1evl(y2, Q0, 8)))
               * 2.50662827463100050242) # ... * sqrt(2pi)
    x = (-2 * log(y)) ** (1/2)
    z = 1/x
    P, Q = (P1, Q1) if x < 8 else (P2, Q2)
    x -= log(x)/x + (z := 1 / x)*polevl(z, P, 8) / p1evl(z, Q, 8)
    return -x if code else x

#----------------------------#

# incbi.c
#----------------------------#
@oideco_C
def invrbeta(yy, aa, bb, *, # inverse regularized incomplete beta function
             R_100=range(100), R_8=range(8)):
    # called `invrbeta(x, a, b)` instead of
    # `betaincinv(a, b, x)`/`incbi(a, b, x)`
    if yy <= 0: return 0
    if yy >= 1: return 1
    x0 = yl = 0
    x1 = yh = 1
    yy_1yy = 1 - yy
    nflg = False
    def exit_f():
        nonlocal x
        if rflg:
            if x <= 2**-53:
                x = 2**-53
            x = 1 - x
        return x
    def under():
        return rflg and 1 - 2**-53
    def ihalve():
        nonlocal a, b, x, x0, x1, y, y0, yp, yh, yl, rflg
        dir_ = 0
        di = 0.5
        for i in R_100:
            if i:
                x = x0 + di*(x1_sx0 := x1 - x0)
                if x == 1:
                    x = 1 - 2**-53
                if x == 0:
                    di = 0.5
                    x = x0 + 0.5*x1_sx0
                    if x == 0:
                        return under()
                y = rbeta(x, a, b)
                if (abs(yp := x1_sx0 / (x1+x0)) < dt
                        or abs(yp := (y-y0) / y0) < dt):
                    return newt()
            if y < y0:
                x0, yl = x, y
                if dir_ < 0:
                    dir_ = 0
                    di = 0.5
                elif dir_ > 3:
                    di += di*(1 - di)
                elif dir_ > 1:
                    di = 0.5*di + 0.5
                else:
                    di = (y0 - y) / (yh - yl)
                dir_ += 1
                if x0 > 0.75:
                    if rflg:
                        rflg = False
                        a, b, y0 = aa, bb, yy
                    else:
                        rflg = True
                        a, b, y0 = bb, aa, yy_1yy
                    y = rbeta(x := 1 - x, a, b)
                    x0 = yl = 0
                    x1 = yh = 1
                    return ihalve()
            else:
                if rflg and (x1 := x) < 2**-53:
                    return under() # not really an underflow therefore
                yh = y             # doesn't look like this in the cephes
                if dir_ > 0:       # source, but conveniently does what
                    dir_ = 0       # we need to do here; we emit no errors
                    di = 0.5       # anyway
                elif dir_ < -3:
                    di *= di
                elif dir_ < -1:
                    di *= 0.5
                else:
                    di = (y - y0) / (yh - yl)
                dir_ -= 1
        if x0 >= 1:
            return exit_f()
        if x <= 0:
            return under()
        return newt()
    def newt():
        nonlocal a, b, nflg, lgm, x, x0, x1, y, yh, yl, d, dt
        if nflg:
            if rflg:
                if x <= 2**-53:
                    x = 2**-53
                x = 1 - x
            return x
        nflg = True
        lgm = lgamma(a + b) - lgamma(a) - lgamma(b)
        for i in R_8:
            if i:
                y = rbeta(x, a, b)
            if y < yl:
                x, y = x0, yl
            elif y > yh:
                x, y = x1, yh
            elif y < y0:
                x0, yl = x, y
            else:
                x1, yh = x, y
            if x in {0, 1}:
                break
            d = (a-1)*log(x) + (b-1)*log(1 - x) + lgm
            if d < -708.396418532264106224:
                return exit_f()
            if d > 709.782712893383996843:
                break
            xt = x - (d := (y-y0) / exp(d))
            if xt <= x0:
                y = (x_sx0 := x-x0) / (x1-x0)
                xt = x0 + 0.5*y*x_sx0
                if xt <= 0:
                    break
            if xt >= x1:
                y = (x1_sx := x1-x) / (x1-x0)
                xt = x1 - 0.5*y*x1_sx
                if xt >= 1:
                    break
            x = xt
            if abs(d / x) < 128 * 2**-53:
                return exit_f()
        dt = 256 * 2**-53
        return ihalve()
    if aa <= 1 or bb <= 1:
        dt = 1e-6
        rflg = False
        a, b, y0 = aa, bb, yy
        y = rbeta(x := a / (a+b), a, b)
    else:
        dt = 1e-4
        yp = -ndtri(yy)
        if yy > 0.5:
            rflg = True
            a, b, y0 = bb, aa, yy_1yy
            yp = -yp
        else:
            rflg = False
            a, b, y0 = aa, bb, yy
        lgm = (yp*yp - 3) / 6
        inv_da_m1 = 1/(2*a - 1)
        inv_db_m1 = 1/(2*b - 1)
        x = 2 / (inv_da_m1 + inv_db_m1)
        d = (yp*(x + lgm)**(1/2) / x
            - (inv_db_m1 - inv_da_m1)
              * (lgm + 5/6 - 2/(3*x)))*2
        if d < -708.396418532264106224:
            return exit_f()
        y = rbeta(x := a / (a + b*exp(d)), a, b)
        if abs(yp := (y-y0) / y0) < 0.2:
            return newt()
    return ihalve()

#----------------------------#

# stdtr.c
#----------------------------#
@oideco_C
def stdtri(confidence, df):
    # called `stdtri(confidence, df)` instead of
    # `stdtri(df, confidence)`
    p = confidence
    k = df
    if k <= 0 or p <= 0 or p >= 1:
        return NAN
    if 0.25 < p < 0.75:
        if p == 0.5:
            return 0
        z = invrbeta(abs(1 - 2*p), 0.5, 0.5*k)
        t = (k*z / (1-z))**(1/2)
        if p < 0.5:
            t = -t
        return t
    rflg = -1
    if p >= 0.75:
        p = 1 - p
        rflg = 1
    if DBL_MAX * (z := invrbeta(2*p, 0.5*k, 0.5)) < k:
        return rflg * INF
    return rflg * (k/z - k)**(1/2)

#----------------------------#
##----------------------------##

@oideco_C
def tcritv(alpha, df, tn=2): # t critical value
    if tn == 2:
        alpha /= 2
    elif tn != 1:
        raise ValueError("invalid tail number") 
    return stdtri(1 - alpha, df)
    

@oideco_C
def cdf(t, v): # cumulative distribution function
    if (t_t := t*t) < v:
        return 1/2 + t*_func_0(v)*ohf(1/2, (v+1)/2, 3/2, -t_t/v)
    return 1 - rbeta(v / (t_t + v), v/2, 1/2)/2

@oideco_C
def tpval(t, df, tp=2): # t-test p-value
    match tp:
        case 0:
            return cdf(t, df)
        case 1:
            return 1 - cdf(t, df)
        case 2:
            return 2 * min(cdf(t, df), 1 - cdf(t, df))
        case 3:
            return 2 * (1 - cdf(abs(t), df))
    raise ValueError("invalid test type")

@oideco
def sstd(arr): # sample stddev
    n = len(arr)
    arr_mn = mean(arr)
    return (sum([(arr_i - arr_mn)**2 for arr_i in arr]) / (n - 1)) ** (1/2)

@oideco
def pstd(arr): # population stddev
    n = len(arr)
    arr_mn = mean(arr)
    return (sum([(arr_i - arr_mn) ** 2 for arr_i in arr]) / n) ** (1/2)

@oideco
def pvar(*arrs): # pooled variance / pooled stddev
    l_arr = [*map(len, arrs)]
    return (sum([(l - 1) * sstd(a) ** 2
                  for l, a in zip(l_arr, x)])
           / (sum(l_arr) - len(l_arr))) ** (1/2)

@oideco(2)
def otstat(arr, apmean): # one-sample t-test
    n = len(arr)
    return (mean(arr) - apmean) * n**(1/2) / sstd(arr), n - 1

@oideco(2)
def ptstat(arr1, arr2): # paired samples t-test
    diffs = [*map(sub, arr1, arr2)]
    l_diffs = len(diffs)
    return mean(diffs) * l_diffs**(1/2) / sstd(diffs), l_diffs - 1

@oideco(2)
def itstat(arr1, arr2, eva=True): # independent samples t-test /
                                  # unpaired samples t-test /
                                  # two-sample t-test
    mn_diff = mean(arr1) - mean(arr2)
    l_a = len(arr1)
    l_b = len(arr2)
    if eva:
        return mn_diff / (pvar(arr1, arr2) * (1/l_a + 1/l_b)**(1/2)), l_a + l_b - 2
    sl_a = sstd(arr1)**2 / l_a
    sl_b = sstd(arr2)**2 / l_b
    return (mn_diff / (sl_a_psl_b := sl_a + sl_b)**(1/2),
            sl_a_psl_b**2 / (sl_a**2/(l_a - 1) + sl_b**2/(l_b - 1)))

if __name__ == '__main__':
    from math import ceil, sqrt
    from traceback import print_exception
    ns = {x: eval(x)
          for x in __all__ + ['exp', 'gamma', 'lgamma',
                              'log', 'int', 'ceil', 'sqrt',
                              'sum']}
    glob = {'__builtins__': {}}
    while True:
        try:
            if t := input('>> '):
                a = eval(t, glob, ns)
                if a is not None:
                    print(repr(a))
        except EOFError as e:
            print_exception(type(e), e, e.__traceback__)
            break
        except (Exception, KeyboardInterrupt) as e:
            print_exception(type(e), e, e.__traceback__)
