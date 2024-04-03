import numpy as np
import scipy
from scipy import io, signal

from algorithm import CCA, FBCCA, TRCA, TDCA, HDCA, BLDA, SVM
from functions import numba_eig, filter_d, pre_process, pca_white, get_template_list


def get_tdcs_filter(train_data, latency):
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
        [eigenvalues, eigenvectors] = scipy.linalg.eig(sb, sw)
        # [eigenvalues, eigenvectors] = numba_eig(sb, sw)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, sorted_indices]


def get_train_test_data(person_id, _start, _end, chan=(55, 47, 54, 56, 53, 57, 60, 61, 62), k_fold=6, pre_filter=True):
    _train_data_list = []
    _test_data_list = []
    if person_id:
        raw_data = io.loadmat(f"./benchmark_dataset/S{person_id}.mat")['data'][chan, ...].transpose(3, 2, 0, 1)
        if pre_filter:
            raw_data = pre_process(raw_data, 250)  # 通带滤波和工频滤波
        for i in range(k_fold):
            train_idx = np.arange(6).tolist()
            test_idx = train_idx.pop(i)
            _train_data_list.append(raw_data[train_idx, :, :, _start: _end])
            _test_data_list.append(raw_data[test_idx, :, :, _start: _end])
    else:
        for i in range(1, 36):
            temp = get_train_test_data(i, _start, _end, chan, k_fold)
            _train_data_list.extend(temp[0])
            _test_data_list.extend(temp[1])

    return _train_data_list, _test_data_list


def projection_np(data_vec, filter_matrix, filter_matrix_inv=None):
    if filter_matrix_inv is None:
        return data_vec @ filter_matrix
    else:
        return data_vec @ filter_matrix @ filter_matrix_inv


def get_projection_data(args):
    train_data, test_data, latency, cr = args
    _filter = get_tdcs_filter(train_data, latency)

    fiter_matrix = _filter[:, :int(cr * 0.01 * _filter.shape[0])]
    try:
        fiter_matrix_inv = scipy.linalg.pinv(fiter_matrix)
    except np.linalg.LinAlgError:
        try:
            fiter_matrix_inv = np.linalg.pinv(fiter_matrix)
        except np.linalg.LinAlgError:
            fiter_matrix_inv = fiter_matrix.T @ scipy.linalg.inv(fiter_matrix @ fiter_matrix.T)
    filtered_train_data = projection_np(np.concatenate(train_data.T).T[..., np.newaxis, :], fiter_matrix, fiter_matrix_inv).reshape(
            train_data.swapaxes(-1, -2).shape).swapaxes(-1, -2)
    filtered_test_data = projection_np(np.concatenate(test_data.T).T[..., np.newaxis, :], fiter_matrix, fiter_matrix_inv).reshape(
            test_data.swapaxes(-1, -2).shape).swapaxes(-1, -2)
    return filtered_train_data, filtered_test_data


def worker(args):
    train_data, test_data, alg = args
    if alg[1]:
        alg[0].init()
        alg[0].train(train_data)
    match len(test_data.shape):
        case 2:
            return alg[0].fit(test_data)[0]
        case 3:
            return [alg[0].fit(a)[0] for a in test_data]
        case 4:
            return [[alg[0].fit(a)[0] for a in b] for b in test_data]


def test():
    freq_set = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2,
                8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4,
                8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6,
                8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8]
    start_loc = 160  # 数据起始位置（0.5s + 0.14s）
    data_len = 250  # 数据长度
    latency = 3  # 自相关矩阵的最大阶次 The maximum order of the autocorrelation matrix
    cr = 10  # 压缩比 (%) compression ratio (%)
    person_id = 1

    train_data_list, test_data_list = get_train_test_data(person_id, start_loc, start_loc + data_len)
    data_list = []
    for i in range(len(train_data_list)):
        data_list.append(get_projection_data((train_data_list[i], test_data_list[i], latency, cr)))

    alg_list = [
        (CCA(freq_set), False),
        (TRCA(), True),
        (TDCA(frequency_set=freq_set, sample_rate=250), True),
    ]

    records = []
    for i in range(len(data_list)):
        for alg in alg_list:
            records.append(worker((data_list[i][0], data_list[i][1], alg)))
    records = np.asarray(records)
    acc_list = np.asarray([np.sum(i == np.arange(1, 41)) / 40 for i in records])
    print(acc_list)
    # result:
    # [0.35 0.875 0.875 0.075 0.825 0.7 0.225 0.85 0.925 0.375 0.925 0.825 0.2 0.95 0.85 0.375 0.975 0.925]


if __name__ == "__main__":
    test()
