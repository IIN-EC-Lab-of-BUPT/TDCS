import numpy as np
import scipy
from numba import prange, njit
from scipy import signal
from scipy.signal import lfilter, lfilter_zi, resample_poly
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca_white(data, n_components, whiten=True, keep_shape=False):
    scaler = StandardScaler()
    pca = PCA(n_components=n_components, whiten=whiten)

    def _f(_data):
        if len(_data.shape) > 2:
            return [_f(d) for d in _data]
        else:
            x_scaled = scaler.fit_transform(_data.T)
            if keep_shape:
                return pca.inverse_transform(pca.fit_transform(x_scaled)).T
            else:
                return pca.fit_transform(x_scaled).T

    return np.asarray(_f(data))


# 预处理
def pre_process(data, srate):
    data = filter_g(data, srate)
    data = filter_d(data, srate)
    return data


# 工频滤波
def filter_g(data, srate):
    fs = srate
    f0 = 50
    q = 35
    b, a = signal.iircomb(f0, q, 'notch', fs=fs)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


# 带通滤波
def filter_d(data, srate, wp=(6, 90), ws=(2, 100)):
    fs = srate / 2
    wp = [wp[0] / fs, wp[1] / fs]
    ws = [ws[0] / fs, ws[1] / fs]
    # N, Wn = signal.cheb1ord(wp, ws, 3, 45)
    # filter_b, filter_a = signal.cheby1(N, 0.5, Wn, btype='bandpass')

    n, wn = signal.ellipord(wp, ws, 3, 40)
    [filter_b, filter_a] = signal.ellip(n, 1, 40, wn, 'bandpass')

    filtered_data = signal.filtfilt(filter_b, filter_a, data, axis=-1)
    return filtered_data


def butter_bandpass_filter(data, low_cut, high_cut, fs, order=4):
    """
    Design band pass filter.

    Args:
        - low_cut  (float) : the low cutoff frequency of the filter.
        - high_cut (float) : the high cutoff frequency of the filter.
        - fs       (float) : the sampling rate.
        - order      (int) : order of the filter, by default defined to 4.
    """
    # calculate the Nyquist frequency
    nyq = 0.5 * fs

    # design filter
    low = low_cut / nyq
    high = high_cut / nyq
    [b, a] = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)


@njit(parallel=True)
def numba_eig(a, b):
    b_inv = np.linalg.inv(b)
    temp = b_inv @ a
    temp = np.asarray(temp, dtype=np.complex128)
    return np.linalg.eig(temp)
    # return scipy.linalg.eig(a, b)


def sparse_eigs(a, b):
    a = scipy.sparse.csc_matrix(a)
    b = scipy.sparse.csc_matrix(b)
    return scipy.sparse.linalg.eigs(a, k=b.shape[0] // 2, M=b)


@njit(parallel=True)
def numba_mean_4(a, axis=None):
    if axis == 0:
        res = np.zeros(a.shape[1:])
        for i in prange(a.shape[0]):
            res += a[i]

        res /= a.shape[0]
        return res


@njit(parallel=True)
def numba_mean_2(a, axis=1):
    if axis == 0:
        res = np.zeros(a.shape[1:])
        for i in prange(a.shape[0]):
            res += a[i]
        return res / a.shape[0]
    elif axis == 1:
        res = np.zeros(a.shape[0])
        for i in prange(a.shape[1]):
            res += a[:, i]
        return res / a.shape[1]


@njit(parallel=True)
def numba_st(raw_data, latency, ch_out, split_len=50):
    filter_list = []
    filter_inv_list = []
    for k in prange(int(raw_data.shape[-1] / split_len)):
        train_data = raw_data[..., k * split_len: k * split_len + split_len]

        nb, nt, nch, data_len = train_data.shape
        new_shape = [x + y for x, y in zip(train_data.shape[::-1], (latency, latency * train_data.shape[-2], 0, 0))][
                    ::-1]
        new_shape = (new_shape[0], new_shape[1], new_shape[2], new_shape[3])
        data = np.zeros(shape=new_shape, dtype=np.float64)
        for i in prange(latency + 1):
            data[..., i * nch:(i + 1) * nch, i: i + data_len] = train_data

        data_a = data
        data_mc = np.zeros(data_a.shape[1:])
        data_mc = numba_mean_4(data_a, axis=0)
        temp = numba_mean_4(data_mc, axis=0)
        hb = 1 / np.sqrt(data_a.shape[1]) * np.hstack([data_mc[i, ...] - temp for i in prange(len(data_mc))])
        hw = np.hstack([data_a[i, j, ...] - numba_mean_2(data_a[i, j, ...], axis=1).reshape((data_a.shape[-2], 1))
                        for i in prange(data_a.shape[0])
                        for j in prange(data_a.shape[1])]) * 1 / np.sqrt(data_a.shape[0] * data_a.shape[1])
        rb = (hb @ hb.T)
        rw = (hw @ hw.T)
        sw = np.zeros((nch * data_len, nch * data_len))
        sb = np.zeros((nch * data_len, nch * data_len))
        for i in prange(data_len):
            for j in prange(data_len):
                if abs(i - j) < latency + 1:
                    ii, jj = i - min(i, j), j - min(i, j)
                    sb[i * nch: (i + 1) * nch, j * nch: (j + 1) * nch] = rb[ii * nch: (ii + 1) * nch,
                                                                         jj * nch: (jj + 1) * nch]
                    sw[i * nch: (i + 1) * nch, j * nch: (j + 1) * nch] = rw[ii * nch: (ii + 1) * nch,
                                                                         jj * nch: (jj + 1) * nch]
        b_inv = np.linalg.inv(sw)
        temp = b_inv @ sb
        temp = np.asarray(temp, dtype=np.complex128)
        eigenvalues, eigenvectors = np.linalg.eig(temp)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        out = ch_out * data_len // 10
        filter_list.append(eigenvectors[:, sorted_indices[:int(data_len * 0.5) + out]])
        filter_inv_list.append(np.linalg.pinv(eigenvectors[:, sorted_indices[:int(data_len * 0.5) + out]]))
    return filter_list, filter_inv_list


def trca(train_data, n=1):
    train_data = np.asarray(train_data)
    (n_block, n_channel, n_point) = train_data.shape
    s = np.zeros((n_channel, n_channel))
    for i in range(0, n_block - 1):
        x1 = train_data[i, :, :]
        x1 = x1 - np.mean(x1, axis=1).reshape(n_channel, 1)
        for j in range(i + 1, n_block):
            x2 = train_data[j, :, :]
            x2 = x2 - np.mean(x2, axis=1).reshape(n_channel, 1)
            s = s + np.dot(x1, x2.transpose()) + np.dot(x2, x1.transpose())
    ux = np.swapaxes(train_data, 0, 1).reshape(n_channel, n_point * n_block)
    ux = ux - np.mean(ux, axis=1).reshape(n_channel, 1)
    try:
        [eigenvalues, eigenvectors] = scipy.linalg.eigh(s, ux @ ux.T)
    except np.linalg.LinAlgError:
        [eigenvalues, eigenvectors] = scipy.linalg.eig(s, ux @ ux.T)
        # [eigenvalues, eigenvectors] = numba_eig(s, ux @ ux.T)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, sorted_indices[:n]]


def tdca(data_a, ch_out=0):
    data_a = data_a - np.mean(data_a, axis=-1, keepdims=True)
    data_mc = np.mean(data_a, axis=0)
    temp = np.mean(data_mc, axis=0)
    hb = np.hstack([data_mc[i, ...] - temp for i in range(len(data_mc))])
    hw = np.hstack([data_a[i, j, ...] - np.mean(data_a[i, j, ...], axis=1).reshape((data_a.shape[-2], 1))
                    for i in range(data_a.shape[0])
                    for j in range(data_a.shape[1])])

    hw = np.hstack(np.vstack(data_a))
    sb = hb @ hb.T * data_a.shape[0]
    sw = hw @ hw.T - sb - temp @ temp.T * data_a.shape[0] * data_a.shape[1]
    try:
        return scipy.linalg.eigh(sb, sw)[1][:, :-ch_out - 1:-1]
    except np.linalg.LinAlgError:
        [eigenvalues, eigenvectors] = scipy.linalg.eig(sb, sw)
        # [eigenvalues, eigenvectors] = numba_eig(sb, sw)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        return eigenvectors[:, sorted_indices[:ch_out]]


def extend_data(data, latency, raw=True, frequency_set=None, sample_rate=None):
    data -= np.mean(data, axis=-1, keepdims=True)
    if raw:
        temp = np.zeros([x * y for x, y in zip(data.shape[::-1], (1, latency + 1, 1, 1))][::-1], dtype=data.dtype)
        for i in range(latency + 1):
            temp[..., i * data.shape[-2]:(i + 1) * data.shape[-2], :data.shape[-1] - i] = data[..., i:]
    else:
        temp = np.zeros([x + y for x, y in zip(data.shape[::-1], (latency, latency * data.shape[-2], 0, 0))][::-1],
                        dtype=data.dtype)
        for i in range(latency + 1):
            temp[..., i * data.shape[-2]:(i + 1) * data.shape[-2], i: i + data.shape[-1]] = data
    if sample_rate:
        projection_matrix_list = [i @ i.T for i in get_template_list(frequency_set, temp.shape[-1], sample_rate)]
        if len(frequency_set) > 1:
            data_p = np.asarray(
                [temp[:, i, ...] @ projection_matrix_list[i] for i in range(len(projection_matrix_list))]).swapaxes(
                0, 1)
            data_a = np.concatenate((temp, data_p), axis=-1)
        else:
            data_a = np.concatenate((temp, temp @ projection_matrix_list[0]), axis=-1)
    else:
        data_a = temp
    return data_a


def cca(data, template):
    data = data.T
    # qr分解,data:length*channel
    q_temp = np.linalg.qr(data)[0]

    template = template.T
    q_cs = np.linalg.qr(template)[0]
    data_svd = np.dot(q_temp.T, q_cs)
    [u, s, v] = np.linalg.svd(data_svd)
    weight = [1.25, 0.67, 0.5]
    rho = sum(s[:3] * weight[:len(s[:3])])
    return rho


def cca_q(q_temp, target_list, k):
    weight = [1.25, 0.67, 0.5]
    res = np.zeros(len(target_list))
    for idx in range(len(target_list)):
        data_svd = np.dot(q_temp.T, target_list[idx])
        s = np.linalg.svd(data_svd)[1]
        a = 0
        for i in range(min(len(s), 3)):
            a += s[i] * weight[i]
        res[idx] = k * a * a
    return res


def get_filter(wp, ws, sample_rate=250, output='ba'):
    wp = np.asarray(wp)
    ws = np.asarray(ws)
    if output == 'ba':
        fs = sample_rate / 2
        n, wn = signal.cheb1ord(wp / fs, ws / fs, 3, 45)
        [filter_b, filter_a] = signal.cheby1(n, 0.5, wn, btype='bandpass')
        return filter_b, filter_a
    elif output == 'sos':
        sos = signal.cheby1(15, 0.5, wp, btype='bandpass', output="sos", fs=sample_rate)
        return sos


def get_template_list(frequency_set, data_len, sample_rate=250, set_phase=True, multi_times=5, qr=True):
    if set_phase:
        phase_set = [i % 4 * 0.5 for i in range(len(frequency_set))]
    else:
        phase_set = [0] * len(frequency_set)

    n = np.arange(0, data_len) / sample_rate
    if qr:
        target_list = np.zeros((len(frequency_set), data_len, multi_times * 2))
    else:
        target_list = np.zeros((len(frequency_set), multi_times * 2, data_len))
    raw = np.zeros((multi_times * 2, data_len))
    for i in range(len(frequency_set)):
        for j in range(multi_times):
            raw[j * 2] = np.cos((j + 1) * frequency_set[i] * np.pi * 2 * n + phase_set[i] * np.pi)
            raw[j * 2 + 1] = np.sin((j + 1) * frequency_set[i] * np.pi * 2 * n + phase_set[i] * np.pi)
        if qr:
            target_list[i] = np.linalg.qr(raw.T)[0]
        else:
            target_list[i] = raw
    return target_list


def horizontal_expanse_to_tense(x, order):
    n = x.shape
    if np.mod(n[1], order):
        print('error(维度不匹配)')
    y = x.reshape(n[0], int(n[1] / order), int(order), order="F")
    return y


def tense_horizontal_expanse(x):
    n = x.shape
    y = x.reshape(n[0], n[1] * n[2], order='F')
    return y


def arord(R, m, mcor, ne, pmin, pmax):
    imax = pmax - pmin

    sbc = np.zeros((1, imax + 1))
    fpe = np.zeros((1, imax + 1))
    logdp = np.zeros((1, imax + 1))
    np_m = np.zeros((1, imax + 1))

    np_m[0, imax] = m * pmax + mcor

    R22 = R[int(np_m[0, imax]):int(np_m[0, imax]) + m, int(np_m[0, imax]):int(np_m[0, imax]) + m]

    invR22 = np.linalg.inv(R22)
    Mp = np.dot(invR22, invR22.T)

    logdp[0, imax] = 2 * np.log(np.abs(np.prod(np.diag(R22))))

    i = imax
    for p in range(pmax, pmin - 1, -1):
        np_m[0, i] = m * p + mcor
        if p < pmax:
            Rp = R[int(np_m[0, i]):int(np_m[0, i]) + m, int(np_m[0, imax]):int(np_m[0, imax]) + m]
            L = np.linalg.cholesky(np.identity(m) + np.dot(np.dot(Rp, Mp), Rp.T)).T
            N = np.dot(np.linalg.inv(L.T), np.dot(Rp, Mp))  # !!!!!!!!!!!!!!!!!
            Mp = Mp - np.dot(N.T, N)
            logdp[0, i] = logdp[0, i + 1] + 2 * np.log(np.abs(np.prod(np.diag(L))))

        sbc[0, i] = logdp[0, i] / m - np.log(ne) * (ne - np_m[0, i]) / ne

        fpe[0, i] = logdp[0, i] / m - np.log(ne * (ne - np_m[0, i]) / (ne + np_m[0, i]))

        i = i - 1

    return sbc, fpe, logdp, np_m


def arqr(v, p, mcor):
    n, m = v.shape

    ne = n - p
    np_m = m * p + mcor
    K = np.zeros((ne, np_m + m))
    if mcor == 1:
        K[:, 0] = np.ones((ne, 1))

    for j in range(1, p + 1):
        K[:, mcor + m * (j - 1):mcor + m * j] = v[p - j:n - j, :]

    K[:, np_m:np_m + m] = v[p:n, :]
    q = np_m + m
    delta = (q ** 2 + q + 1) * np.finfo(np.float64).eps  # !!!!!!!!!!!
    scale = np.sqrt(delta) * np.sqrt(np.sum(np.power(K, 2), axis=0))

    Q, R = np.linalg.qr(np.vstack((K, np.diag(scale))))
    # Q, R = np.linalg.qr(np.vstack((K, np.diag(np.array(scale).squeeze()))), mode='complete')
    R = np.triu(R)

    return R, scale


def arfit(v, pmin, pmax, selector, no_const):
    n, m = v.shape
    if not isinstance(pmin, int) or not isinstance(pmax, int):
        print("error: Order must be integer.")
    if pmax < pmin:
        print("error: PMAX must be greater than or equal to PMIN.")
    if selector is None:
        mcor = 1
        selector = 'sbc'
    elif no_const is None:
        if selector == 'zero':
            mcor = 0
            selector = 'sbc'
        else:
            mcor = 1
    else:
        if no_const == 'zero':
            mcor = 0
        else:
            raise ValueError("Bad argument. Usage:  [A, C] = arfit(v, pmin, pmax, SELECTOR, 'zero')")

    ne = n - pmax
    npmax = m * pmax + mcor

    if ne <= npmax:
        print("Time series too short.")
    R, scale = arqr(v, pmax, mcor)
    sbc, fpe, logdp, notuse = arord(R, m, mcor, ne, pmin, pmax)
    # val = eval(selector).min(0)
    iopt = np.argmin(eval(selector))
    popt = pmin + iopt  # !!!!!!!!!!
    np_m = m * popt + mcor  # !!!!!!!!!!!
    R11 = R[0:np_m, 0:np_m]
    R12 = R[0:np_m, npmax:npmax + m]
    R22 = R[np_m:npmax + m, npmax:npmax + m]
    if np_m > 0:
        if mcor == 1:
            con = scale[1:npmax + m] / scale[0]  # !!!!!!!!!!
            R11[:, 0] = np.dot(R11[:, 0], con)
        Aaug = np.dot(np.linalg.inv(R11), R12).T
        if mcor == 1:
            # w = np.dot(Aaug[:, 0], con)
            A = Aaug[:, 1:np_m]
        else:
            # w = np.zeros((m, 1))
            A = Aaug  # np.zeros((0, 0))
    else:
        # w = np.zeros((m, 1))
        A = np.zeros((0, 0))
    dof = ne - np_m
    C = np.dot(R22.T, R22) / dof
    # invR11 = np.linalg.inv(R11)
    # if mcor == 1:
    #     invR11[0, :] = np.dot(invR11[0, :], con)
    # Uinv = np.dot(invR11, invR11.T)
    # th = np.hstack((np.array(dof), np.zeros((1, Uinv.shape[1] - 1))))
    # th = np.vstack((np.zeros((1, Uinv.shape[1])), Uinv))
    # th[0, 0] = dof
    # return w, A, C, sbc, fpe, th
    order = int(A.shape[-1] / C.shape[-1] + 1)
    return A, C, order


def estimate_ste_equalizer(data, P):
    if P is None:
        P = [4, 6]
    elif len(P) < 2:
        P[1] = P[0]

    # 去均值
    data = data - np.reshape(np.mean(data, 1), [data.shape[0], 1])

    chan_num = data.shape[0]
    parameter_matrix, noise_matrix, equalizerOrder = arfit(data.T, P[0], P[1], 'fpe', 'zero')
    armodel = np.hstack((np.identity(chan_num), -parameter_matrix))
    [noise_value, noise_vector] = np.linalg.eigh(noise_matrix)
    noise_vector = -noise_vector
    noise_vector[:, 3] = -noise_vector[:, 3]
    noise_value = noise_value.reshape(len(noise_value), 1)
    equalizerTense = np.dot(noise_vector.T, armodel)

    STEqualizer = horizontal_expanse_to_tense(
        np.dot(np.diag((1 / np.sqrt(noise_value)).reshape(noise_value.shape[0])), equalizerTense), equalizerOrder)
    return STEqualizer


def ste_filter(filter_cell_matrix, input_data, zi=None):
    mo, mi = filter_cell_matrix.shape[0:2]
    if mi != input_data.shape[0]:
        raise Exception('输入数据与线性系统输入维度不同，线性相位系统维度:', mi, '输入数据维度', input_data.shape[0])

    Zf = np.zeros((mo, mi, filter_cell_matrix.shape[2] - 1))
    output_data = np.zeros((mo, input_data.shape[1]))
    temp_data = np.zeros((mi, input_data.shape[1]))
    for i in range(mo - 1, -1, -1):
        for j in range(mi - 1, -1, -1):
            if zi is not None:
                temp_data[j, :], Zf[i, j, :] = lfilter(filter_cell_matrix[i, j, :], np.array([1]), input_data[j, :],
                                                       zi=zi[i, j])
            else:
                temp_data[j, :], Zf[i, j, :] = lfilter(filter_cell_matrix[i, j, :], np.array([1]), input_data[j, :],
                                                       zi=lfilter_zi(filter_cell_matrix[i, j, :], np.array([1])) *
                                                          input_data[j, 0])
                # zi=np.ones((filter_cell_matrix.shape[2] - 1,)))
        output_data[i, :] = temp_data.sum(0)
    return output_data, Zf


def iterate_qr(r, new_data):
    if r is None:
        g1_size = 0
        r = new_data.T
    else:
        g1_size = r.shape[0]
        r = np.vstack((r, new_data.T))
    #     G,r_new = np.linalg.qr(R, mode='reduced')
    G, r_new, e = scipy.linalg.qr(r, mode='economic', pivoting=True)  # !!!!!!!!!!
    # matlabQR分解会重排矩阵，还原重排结果
    E = np.identity(r_new.shape[1])
    r_new = np.dot(r_new, E[e])
    if g1_size == 0:
        g1 = np.zeros((new_data.shape[0], new_data.shape[0]))
    else:
        g1 = G[0:g1_size, :]
    g2 = G[g1_size:, :]
    return r_new, g1, g2


def new_qr(r, new_data):
    if r is None:
        g1_size = 0
        r = new_data.T
    else:
        g1_size = r.shape[0]
        r = np.vstack((r, new_data.T))
    G, r_new = np.linalg.qr(r)
    if g1_size == 0:
        g1 = np.zeros((new_data.shape[0], new_data.shape[0]))
    else:
        g1 = G[0:g1_size, :]
    g2 = G[g1_size:, :]
    return r_new, g1, g2


@njit(parallel=True)
def update_inner_product(cell, max_window_index, x_g1, x_g2, t_g1, t_g2):
    """
    原地更新内积，不返回值
    :param cell: 待更新的内积数组
    :param max_window_index:
    :param x_g1:
    :param x_g2:
    :param t_g1:
    :param t_g2:
    :return: None
    """
    for i in prange(t_g1.shape[0]):
        for j in range(max_window_index - 1, 0, -1):
            cell[i, j] = np.dot(np.dot(x_g1.T, cell[i, j - 1]), t_g1[i, j]) + np.dot(x_g2.T, t_g2[i, j])
        cell[i, 0] = np.dot(x_g2.T, t_g2[i, 0])


@njit(parallel=True)
def get_result(cell, max_window, step):
    noise_energy_matrix = np.zeros((cell.shape[0], max_window))
    signal_energy_cell = np.zeros((cell.shape[0], max_window))
    snr = np.zeros((cell.shape[0], max_window))
    for i in prange(cell.shape[0]):
        for j in prange(max_window):
            temp = np.dot(cell[i, j], cell[i, j].T)
            eigenvalues = np.linalg.eigvals(temp)
            s = np.sort(eigenvalues)
            noise_energy_matrix[i, j] = (temp.shape[0] - eigenvalues.sum()) * step * (j + 1)
            snr[i, j] = eigenvalues.sum() / (temp.shape[0] - eigenvalues.sum())
            signal_energy_cell[i, j] = 1.25 * s[-1] + 0.67 * s[-2] + 0.5 * s[-3]

    return noise_energy_matrix, signal_energy_cell, snr


def get_stc_filter(train_data, latency):
    nb, nt, nch, data_len = train_data.shape
    data = np.zeros((nb, nt, nch + nch * latency, data_len + latency))
    for i in range(latency + 1):
        data[..., i * nch:(i + 1) * nch, i: i + data_len] = train_data

    data_a = data
    data_mc = np.mean(data_a, axis=0)
    temp = np.mean(data_mc, axis=0)
    hb = 1 / np.sqrt(data_a.shape[1]) * np.hstack([data_mc[i, ...] - temp for i in range(len(data_mc))])
    hw = np.hstack([data_a[i, j, ...] - np.mean(data_a[i, j, ...], axis=1).reshape((data_a.shape[-2], 1))
                    for i in range(data_a.shape[0])
                    for j in range(data_a.shape[1])]) * 1 / np.sqrt(data_a.shape[0] * data_a.shape[1])
    rb = (hb @ hb.T)
    rw = (hw @ hw.T)
    sw = np.zeros((nch * data_len, nch * data_len))
    sb = np.zeros((nch * data_len, nch * data_len))
    for i in range(data_len):
        for j in range(data_len):
            if abs(i - j) < latency + 1:
                ii, jj = i - min(i, j), j - min(i, j)
                sb[i * nch: (i + 1) * nch, j * nch: (j + 1) * nch] = rb[ii * nch: (ii + 1) * nch,
                                                                     jj * nch: (jj + 1) * nch]
                sw[i * nch: (i + 1) * nch, j * nch: (j + 1) * nch] = rw[ii * nch: (ii + 1) * nch,
                                                                     jj * nch: (jj + 1) * nch]
    try:
        [eigenvalues, eigenvectors] = scipy.linalg.eigh(sb, sw)
    except np.linalg.LinAlgError:
        # [eigenvalues, eigenvectors] = scipy.linalg.eig(sb, sw)
        [eigenvalues, eigenvectors] = numba_eig(sb, sw)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, sorted_indices]
