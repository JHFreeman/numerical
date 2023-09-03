import numpy as np
import math

def euler(f, a, b, N = 100, alpha):
    h = (b - a) / N
    t = a
    w = alpha
    for i in range(1, N + 1):
        w = h * f(t, w)
        t = a + i * h
        
    
def runge_kutta_order_four(f, a, b, N = 100, alpha):
    h = (b - a) / N
    t = a
    w = alpha
    for i in range(1, N + 1):
        K1 = h * f(t, w)
        K2 = h * f(t + h / 2., w + K1 / 2.)
        K3 = h * f(t + h / 2., w + K2 / 2.)
        K4 = h * f(t + h, w + K3)
        w = w + (K1 + 2 * K2 + 2 * K3 + K4) / 6.
        t = a + i * h

def runge_kutta_fehlberg(f, a, b, alpha, TOL = 1e-08, hmax, hmin):
    t = a
    w = alpha
    h = hmax
    FLAG = True
    while FLAG:
        K1 = h * f(t, w)
        K2 = h * f(t + 0.25 * h, w + 0.25 * K1)
        K3 = h * f(t + 0.375 * h, w + 0.09375 * K1 + 0.28125 * K2)
        K4 = h * f(t + 12 * h / 13, w[i-1] + 1932 * K1/2197 - 7200 * K2 / 2197 + 7296 * K3 / 2197)
        K5 = h * f(t + h, w + 439 * K1 / 216 - 8 * K2 + 3680 * K3 / 513 - 845 * K4 / 4104)
        K6 = h * f(t + 0.5 * h, w - 8 * K1 / 27 + 2 * K2 - 3544 * K3 / 2565 +1859 * K4 / 4104 - 11 * K5 / 40)
        R = math.abs(K1 / 360 - 128 * K3 / 4275 - 2197 * K4 / 75240 + K5 / 50 + 2 * K6 / 55) / h
        if R <= TOL:
            t = t + h
            w = w + 25 * K1 / 216 + 1408 * K3 / 2565 + 2197 * K4 / 4104 - K5 / 5
        delta = 0.84 * math.pow(TOL / R, 0.25)
        if delta <= 0.1:
            h = 0.1 * h
        elif delta >= 4:
            h = 4 * h
        else:
            h = delta * h
        if h > hmax:
            h = hmax
        if t >= b:
            FLAG = False
        elif t + h > b:
            h = b - t
        elif h < hmin:
            FLAG = False

def adams_fourth_order_predictor_corrector(f, a, b, N = 100, alpha):
    h = (b - a) / N
    t = np.array([a])
    w = np.array([alpha])
    yield t[0], w[0]
    for i in range(1, 4):
        K1 = h * f(t[i-1], w[i-1])
        K2 = h * f(t[i-1] + h * 0.5, w[i-1] + K1 * 0.5)
        K3 = h * f(t[i-1] + h * 0.5, w[i-1] + K2 * 0.5)
        K4 = h * f(t[i-1] + h, w[i-1] + K3)
        w[i] = w[i-1] + (K1 + 2 * K2 + 2 * K3 + K4) / 6
        t[i] = a + i * h
    for i in range(4, N + 1):
        t0 = a + i * h
        w0 = w[3] + h * (55 * f(t[3], w[3]) - 59 * f(t[2], w[2]) + 37 * f(t[1], w[1]) - 9 * f(t[0], w[0])) / 24
        w0 = w[3] + h * (9 * f(t0, w0) + 19 * f(t[3], w[3]) - 5 * f(t[2], w[2]) + f(t[1], w[1])) / 24
        for j in range(0, 3):
            t[j] = t[j + 1]
            w[j] = w[j + 1]
        t[3] = t0
        w[3] = w0
        
def runge_kutta_systems_of_differential_equations(f, a, b, m, N = 100, alpha):
    h = (b - a) / N
    t = a
    w = np.array([])
    for j in range(0, m):
        w[j] = alpha[j]
    for i in range(1, N + 1):
        k = np.matrix([[],[]])
        for j in range(0, m):
            k[0][j] = h * f[j](t,w)
        for j in range(0, m):
            k[1][j] = h * f[j](t,w + 0.5 * k[0])
        for j in range(0, m):
            k[2][j] = h * f[j](t,w + 0.5 * k[1])
        for j in range(0, m):
            k[3][j] = h * f[j](t,w + k[2])
        for j in range(0, m):
            w[j] = w[j] + (k[0][j] + 2 * k[1][j] + 2 * k[2][j] + k[3][j])
        t = a + i * h
        
def trapeziodal_with_newton_iteration(f, f_y, a, b, N = 100, alpha, TOL = 1e-08, M):
    h = (b - a) / N
    t = a
    w = alpha
    for i in range(1, N + 1):
        k1 = w + h * f(t, w) * 0.5
        w0 = k1
        j = 1
        FLAG = False
        while FLAG:
            w = w0 - (w0 - h * f(t + h, w0) * 0.5 - k1) / (1 - h * f_y(t + h, w0))
            if math.abs(w-w0) < TOL:
                FLAG = True
            else:
                j = j + 1
                w0 = w
                if j > M:
                    return
        t = a + i * h

def extrapolation(f, a, b, alpha, TOL, hmax, hmin):
    NK = np.array([2, 4, 6, 8, 12, 16, 24, 32])
    TO = a
    WO = alpha
    h = hmax
    FLAG = 1
    Q = np.array([[]])
    Q.resize((7, 7))
    for i in range(0, 7, 1):
        for j in range(0, i + 1, 1):
            Q[i, j] = (NK[i + 1] / NK[j]) ** 2
    while FLAG == 1:
        k = 0
        NFLAG = 0
        y = np.array([])
        y.resize((8))
        while k < 8 and NFLAGS == 0:
            HK = h / NK[k]
            T = TO
            W2 = WO
            W3 = W2 + HK * f(T, W2)
            for j in range(1, NK[k] - 1):
                W1 = W2 
                W2 = W3 
                W3 = W1 + 2. * HK * f(T, W2)
                T = TO + (j + 1) * HK
            y[k] = (W3 + W2 + HK * f(T, W3)) / 2.
            if k >= 1:
                j = k
                v = y[0]
                while j >= 1:
                    y[j - 1] = y[j] + (y[j] - y[j - 1]) / (Q[k - 1, j - 1] - 1)
                    j -= 1
                if abs(y[0] - v) <= TOL:
                    NFLAG = 1
            k += 1
        k -= 1
        if NFLAG == 0:
            h = h / 2.
            if h < hmin:
                FLAG = 0
        else:
            WO = y[0]
            TO += h
            if TO >= b:
                FLAG = 0
            elif TO + h > b:
                h = b - TO
            elif k <= 3 and h < 0.5 * hmax:
                h = 2. * h
