import numpy as np
# import cupy as cp
cp = np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.svm import SVC

from functions import *


class TDCS:
    def __init__(self, n_components=2, latency=3):
        self.n_components = n_components
        self.latency = latency
        self._fiter = None
        self.signal_space = None
        self.template_matrix = None

    def init(self):
        self._fiter = None
        self.signal_space = None
        self.template_matrix = None

    def train(self, train_data):
        self.signal_space = get_stc_filter(train_data, latency=self.latency)
        self.template_matrix = self._deal_data(train_data)

    def fit(self, test_data):
        cors = [np.corrcoef(self._deal_data(test_data), d)[0, 1] for d in self.template_matrix]
        pre = cors.index(np.max(cors)) + 1
        return pre, cors

    def _deal_data(self, data):
        temp = np.concatenate(data.T).T[..., np.newaxis, :]
        return temp @ self.signal_space[:, :self.n_components * data.shape[-1] * data.shape[-2]]


class FilterBank:
    def __init__(self, filter_num=5, step=8, wp_min=6, wp_max=90):
        self.step = step
        self.wp_min = wp_min
        self.wp_max = wp_max
        wp = [(5, 90), (14, 90), (22, 90), (30, 90), (38, 90)]
        ws = [(3, 92), (12, 92), (20, 92), (28, 92), (36, 92)]
        self._fp_list = [get_filter(wp[k], ws[k], output='sos')
                         for k in range(filter_num)]
        # self._fp_list = [get_filter([wp_min + k * step, self.wp_max], [wp_min + k * step - 2, self.wp_max + 2], output='sos')
        #                  for k in range(filter_num)]

    def _get_fb_data(self, data):
        # return [signal.filtfilt(fb, fa, data) for fb, fa in self._fp_list]
        return [signal.sosfiltfilt(sos, data, axis=-1) for sos in self._fp_list]

    @staticmethod
    def _weight(k):
        return pow(k + 1, -1.25) + 0.25


class TDCA:
    def __init__(self, latency=5, ch_out=8, frequency_set=None, sample_rate=None):
        self.latency = latency
        self.ch_out = ch_out
        self.freq_set = frequency_set
        self.sample_rate = sample_rate

        self.filter = None
        self.transformed_data = None
        self.data_center = None

    def init(self):
        self.filter = None
        self.transformed_data = None
        self.data_center = None

    def train(self, train_data):
        self.filter, self.transformed_data = self._train(train_data)

    def _train(self, train_data):
        data_a = extend_data(train_data, self.latency, frequency_set=self.freq_set, sample_rate=self.sample_rate)
        f = tdca(data_a, self.ch_out)
        data_a = data_a - np.mean(data_a, axis=-1, keepdims=True)
        return f, [f.T @ i for i in np.mean(data_a, axis=0)]

    def fit(self, test_data):
        cors = []
        for i, d in enumerate(self.transformed_data):
            temp = extend_data(test_data, self.latency, frequency_set=[self.freq_set[i]], sample_rate=self.sample_rate)
            cors.append(np.corrcoef((self.filter.T @ temp).reshape(-1), d[:self.ch_out].reshape(-1))[0, 1])
        pre = cors.index(max(cors)) + 1
        return pre, cors


class FBTDCA(FilterBank, TDCA):
    def __init__(self, latency=0, ch_out=8, frequency_set=None, sample_rate=None, filter_num=5, step=8, wp_min=5,
                 wp_max=90):
        FilterBank.__init__(self, filter_num, step, wp_min, wp_max)
        TDCA.__init__(self, latency, ch_out, frequency_set, sample_rate)

        self.filter_list = []
        self.transformed_data_list = []

    def init(self):
        self.filter_list = []
        self.transformed_data_list = []

    def train(self, train_data):
        for data in self._get_fb_data(train_data):
            result = self._train(data)
            self.filter_list.append(result[0])
            self.transformed_data_list.append(result[1])

    def fit(self, test_data):
        result = np.zeros((len(self.transformed_data_list), len(self.transformed_data_list[0])))
        for k, data in enumerate(self._get_fb_data(test_data)):
            for i, transformed_data in enumerate(self.transformed_data_list[k]):
                temp = extend_data(data, self.latency, frequency_set=[self.freq_set[i]], sample_rate=self.sample_rate)
                # rho = cca(self.filter_list[k].T @ data, transformed_data)
                rho = np.corrcoef((self.filter_list[k].T @ temp).reshape(-1),
                                  transformed_data[:self.ch_out].reshape(-1))[0, 1]
                result[k, i] = self._weight(k) * rho
        result = np.sum(result, axis=0)
        return result.argmax() + 1, result.tolist()


class CCA(object):
    def __init__(self, frequency_set, data_len=None, template_list=None):
        self.frequency_set = frequency_set
        if data_len:
            self.target_list = get_template_list(np.asarray(self.frequency_set), data_len)
        elif template_list is not None:
            self.target_list = template_list
        else:
            self.target_list = []

    def fit(self, data):
        if len(self.target_list) == 0:
            self.target_list = get_template_list(np.asarray(self.frequency_set), data.shape[-1])
        p = []
        for template in self.target_list:
            rho = cca(data, np.asarray(template)[:, :data.shape[1]].T)
            p.append(rho)
        result = p.index(max(p))
        result = result + 1
        return result, p


class FBCCA(CCA):
    def __init__(self, frequency_set, data_len=None, filter_num=5, step=8, wp_min=6, wp_max=90, template_list=None):
        super().__init__(frequency_set, data_len, template_list)
        self.step = step
        self.wp_min = wp_min
        self.wp_max = wp_max
        self.wp_list = [wp_min + i * step for i in range(filter_num)]

    def fit(self, data):
        if len(self.target_list) == 0:
            self.target_list = get_template_list(np.asarray(self.frequency_set), data.shape[-1])
        cor_u = np.zeros(len(self.target_list))
        for k in range(len(self.wp_list)):
            p1 = [self.wp_list[k], self.wp_max]
            s1 = [self.wp_list[k] - 2, self.wp_max + 10]
            fb, fa = get_filter(p1, s1)
            data_temp = signal.filtfilt(fb, fa, data)
            q_temp = np.linalg.qr(data_temp.T)[0]
            cor_u += cca_q(q_temp, self.target_list, np.power(k + 1, -1.25) + 0.25)

        predict = np.argmax(cor_u) + 1
        return predict, cor_u


class TRCA(object):
    def __init__(self):
        self.wn_t = []
        self.aver_data = []

    def init(self):
        self.wn_t = []
        self.aver_data = []

    def train(self, train_data):
        self.aver_data = np.mean(train_data, axis=0)
        for n_target in range(train_data.shape[1]):
            wn = trca(train_data[:, n_target, :, :])
            self.wn_t.append(wn.transpose())
        self.wn_t = np.asarray(self.wn_t)

    def fit(self, test_data):
        try:
            cors = []
            for i in range(len(self.wn_t)):
                x1 = self.wn_t[i] @ self.aver_data[i, :, :test_data.shape[-1]]
                x2 = self.wn_t[i] @ test_data[:, :]
                cors.append(np.corrcoef(x1, x2)[0, 1])
            pre = cors.index(max(cors)) + 1
            return pre, cors
        except AttributeError:
            print("请先训练模型")


class FBTRCA(object):
    def __init__(self, filter_num=5, step=8, wp_min=6, wp_max=90):
        self.step = step
        self.wp_min = wp_min
        self.wp_max = wp_max
        self.wp_list = [wp_min + i * step for i in range(filter_num)]
        self.wn_t = []
        self.aver_data = []

    def init(self):
        self.wn_t = []
        self.aver_data = []

    def train(self, train_data):
        self.aver_data = np.mean(train_data, axis=0)
        for n_target in range(train_data.shape[1]):
            data = np.squeeze(train_data[:, n_target, :, :])
            fb_wn = []
            for k in range(len(self.wp_list)):
                p1 = [self.wp_list[k], self.wp_max]
                s1 = [self.wp_list[k] - 2, self.wp_max + 5]
                fb, fa = get_filter(p1, s1)
                flit_data = signal.filtfilt(fb, fa, data)
                wn = trca(flit_data)
                fb_wn.append(wn.transpose())
            self.wn_t.append(fb_wn)
        self.wn_t = np.asarray(self.wn_t)

    def fit(self, test_data):
        if isinstance(self.aver_data, np.ndarray) and isinstance(self.wn_t, np.ndarray):
            cors = []
            for i in range(self.aver_data.shape[0]):
                cor_fb = 0
                for k in range(self.wn_t.shape[1]):
                    x1 = np.dot(self.wn_t[i, k, :], self.aver_data[i, :, :])
                    x2 = np.dot(self.wn_t[i, k, :], test_data[:, :])
                    cor1 = np.corrcoef(x1, x2)[0, 1]
                    cor_fb = cor_fb + (pow(k + 1, -1.25) + 0.25) * pow(cor1, 2)
                cors.append(cor_fb)
            pre = cors.index(max(cors)) + 1
            return pre, cors
        else:
            raise ValueError("算法未训练")


class MultilayerWindowClassifier:
    def __init__(self):
        self.channels_number = None
        self.window_step = None
        self.max_window_length = None
        self.max_window_num = None
        self.equalizer_order = None
        self.template_set = None
        self.template_G1 = None
        self.template_G2 = None
        self.spatio_temporal_equalizer = None
        self.QxQy_inner_product = None
        self.current_idx = None
        self.reach_full_window_flag = None
        self.equalized_data_zf = None
        self.equalized_data_R = None
        self.detect_win_num = None
        self.check_win_num = None
        self.threshold = None

    def initial(self, parameters):
        self.window_step = parameters.window_step
        self.max_window_length = parameters.max_window_length
        self.max_window_num = int(self.max_window_length / self.window_step)
        self.channels_number = parameters.channels_number
        self.equalizer_order = parameters.equalizer_order
        self.set_template_set(parameters.template_set)
        self.detect_win_num = parameters.detect_win_num
        self.check_win_num = parameters.check_win_num
        self.threshold = np.copy(parameters.threshold)

        self.clear()

    def update_equalizer(self, noise_data):
        self.spatio_temporal_equalizer = estimate_ste_equalizer(noise_data, self.equalizer_order)

    def add_data(self, new_data):
        if new_data.shape[1] != self.window_step:
            print('error([输入数据长度({})与预定窗长({})不匹配]'.format(new_data.shape[1], self.window_step))

        # equalized_new_data = new_data
        # 逐数据块去均值(整体去均值无法迭代)
        new_data = new_data - np.reshape(np.mean(new_data, axis=1), (new_data.shape[0], 1))
        equalized_new_data, self.equalized_data_zf = ste_filter(self.spatio_temporal_equalizer, new_data,
                                                                self.equalized_data_zf)
        self.equalized_data_R, x_G1, x_G2 = iterate_qr(self.equalized_data_R, equalized_new_data)

        update_inner_product(self.QxQy_inner_product, self.current_idx, x_G1, x_G2, self.template_G1, self.template_G2)

        if self.current_idx < self.max_window_num:
            self.current_idx = self.current_idx + 1
            self.reach_full_window_flag = False
        else:
            self.reach_full_window_flag = True

    def get_result(self):
        if not self.reach_full_window_flag:
            max_window = self.current_idx - 1
        else:
            max_window = self.current_idx

        noise_energy, signal_energy, snr = get_result(self.QxQy_inner_product, max_window, self.window_step)
        result = 0
        length = 0
        result_vector = np.zeros(max_window, dtype=int)
        for i in range(max_window):
            judge2 = pd.Series(noise_energy[:, i]).kurt() >= self.threshold[1]
            judge3 = pd.Series(signal_energy[:, i]).kurt() >= self.threshold[1]
            if judge3 or judge2:
                # result_vector[i] = np.argmin(noise_energy[:, i]) + 1
                # result_vector[i] = np.argmax(signal_energy[:, i]) + 1
                result_vector[i] = np.argmax(signal_energy[:, i] / noise_energy[:, i]) + 1

            if i + 1 >= self.detect_win_num:
                counter = np.bincount(result_vector[i + 1 - self.detect_win_num:i + 1])
                if np.argmax(counter) > 0 and max(counter) >= self.check_win_num:
                    result = np.argmax(counter)
                    length = i + 1

        return result, length

    def clear(self):
        self.reach_full_window_flag = False

        self.QxQy_inner_product = np.zeros((40, self.max_window_num, self.channels_number,
                                            self.template_set.shape[1]))
        self.current_idx = 1

        # 初始化均衡数据参数
        self.equalized_data_zf = None
        self.equalized_data_R = None

    def set_spatio_temporal_equalizer(self, spatio_temporal_equalizer):
        self.spatio_temporal_equalizer = spatio_temporal_equalizer

    def set_template_set(self, template_set):
        if self.template_set is None:
            self.template_set = template_set

        # 计算模板QR分解
        self.template_G1 = np.zeros((template_set.shape[0], self.max_window_num, self.template_set.shape[1],
                                     template_set.shape[1]))
        self.template_G2 = np.zeros((template_set.shape[0], self.max_window_num, self.window_step,
                                     template_set.shape[1]))

        for template_index in range(0, len(self.template_set)):
            template = self.template_set[template_index]
            template = template[:, 0:self.max_window_length]
            template_extend = np.reshape(np.asarray(template),
                                         (template.shape[0], self.window_step, self.max_window_num), order='F')
            r = None
            for timestep_index in range(0, self.max_window_num):
                r, tg1, tg2 = iterate_qr(r, template_extend[..., timestep_index])
                self.template_G1[template_index, timestep_index] = tg1
                self.template_G2[template_index, timestep_index] = tg2


class HDCA:
    def __init__(self, threshold=0, window_length=15, stride=0.5, window_num=None):
        self.lda_list = []
        self.second_model = LogisticRegression()
        self.weight_list = []
        if window_length:
            self.window_length = window_length
            self.window_num = None
            self.step = int(window_length * stride)
        else:
            self.window_length = None
            self.window_num = window_num
            self.step = None
        self.stride = stride
        self.n_components = 30
        self.scaler_list = []
        self.pca_filter = None
        self.threshold = threshold

    def init(self):
        self.second_model = LogisticRegression()
        # self.second_model = svm.SVC(kernel='linear', class_weight='balanced')
        self.lda_list = []
        self.weight_list = []

    def train(self, train_data, train_label):
        """
        Input: EEG Signal ∈ R^(D×T×N)(D channels, T samples, and N image number)
        """
        # train_data = pca_white(train_data, self.n_components)
        # train_data = train_data - np.mean(train_data, axis=-1, keepdims=True)
        # target_data = train_data[np.where(y == 1)[0]].transpose(1, 2, 0)
        # no_target_data = train_data[np.where(y == 2)[0]].transpose(1, 2, 0)
        # x1_label = np.ones((1, target_data.shape[2]))
        # x2_label = np.ones((1, no_target_data.shape[2])) * 2
        # train_label = np.concatenate((x1_label, x2_label), axis=1)
        # train_label = np.squeeze(train_label)

        # for i in range(train_data.shape[1]):  # iterate over channels
        #     scaler = StandardScaler()
        #     train_data[:, i, :] = scaler.fit_transform(train_data[:, i, :])
        #     self.scaler_list.append(scaler)
        # data_reshaped = train_data.transpose(0, 2, 1).reshape(-1, train_data.shape[1])
        # pca = PCA()
        # pca.fit(data_reshaped)
        # self.pca_filter = pca.components_[:self.n_components]
        # # 应用空间滤波器 - 得到形状为 (trials, n_components, samples) 的数组
        # train_data = np.einsum('ij,kjl->kil', self.pca_filter, train_data)

        # 未来可以增加鲁棒性
        if self.step is None:
            self.step = int(train_data.shape[-1] / self.window_num * self.stride)
        start = np.arange(0, train_data.shape[-1] - self.step, self.step, dtype=int)
        y_i = np.zeros((train_label.shape[0], len(start)))
        for i in range(len(start)):
            self.lda_list.append(LinearDiscriminantAnalysis())
            # temp = np.dstack(
            #     (target_data[:, start[i]: start[i] + step, :], no_target_data[:, start[i]: start[i] + step, :]))
            temp = train_data[:, :, start[i]: start[i] + self.step]
            temp = np.mean(temp, axis=-1)
            # the shape of train data is (N, D), the shape of label is (N, )
            try:
                self.lda_list[i].fit(temp, train_label)
            except ValueError:
                self.lda_list[i] = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                self.lda_list[i].fit(temp, train_label)
            self.weight_list.append(self.lda_list[i].coef_)
            y_i[:, i] = np.squeeze(np.dot(temp, self.weight_list[i].transpose()))

        # 第二步
        # the shape of train data is (N, i), the i is the number of time windows, the shape of label should be (N, )
        self.second_model.fit(y_i, train_label)

    def predict(self, data):
        """
        判决
        :param data: 脑电数据，shape=(D channels, T samples, and N image number)
        :return: 识别结果和决策值
        """
        # data = pca_white(data, self.n_components)
        # data = data - np.mean(data, axis=-1, keepdims=True)
        # for i in range(data.shape[1]):  # iterate over channels
        #     data[:, i, :] = self.scaler_list[i].fit_transform(data[:, i, :])
        # 应用空间滤波器 - 得到形状为 (trials, n_components, samples) 的数组
        # data = np.einsum('ij,kjl->kil', self.pca_filter, data)

        data = self._deal_data(data)
        # result = self.second_model.predict(data)
        judgement = self.second_model.decision_function(data)
        result = (judgement > self.threshold) + 1
        return result, judgement

    def get_score(self, data, label):
        """
        获取得分（AUC）
        :param data: 脑电数据，shape=(D channels, T samples, and N image number)
        :param label: 标签，应为一维向量
        :return:（float）AUC
        """
        result, judgement = self.predict(data)
        # print(accuracy_score(label, result))
        auc = roc_auc_score(label, judgement)
        f1 = f1_score(label, result)
        return auc, f1, result

    def _deal_data(self, data):
        data = np.asarray(data)
        start = np.arange(0, data.shape[-1] - self.step, self.step, dtype=int)
        cache = [np.mean(data[:, :, start[i]: start[i] + self.step], axis=-1) @ self.weight_list[i].T
                 for i in range(len(start))]
        cache = np.asarray(cache)
        cache = np.squeeze(cache)
        cache = cache.transpose()
        return cache


class SVM:
    def __init__(self):
        # 在这里我们使用SVC，因为EEG数据是分类问题
        # ‘class_weight’参数设置为‘balanced’，这将根据类别中的样本数量自动调整权重，从而处理类别不平衡的问题
        self.model = SVC(kernel='rbf', class_weight='balanced')
        data_len = 150
        self.window_len = 15
        step = int(self.window_len * 0.5)
        self.start = np.arange(0, data_len - step, step, dtype=int)

    def init(self):
        self.model = SVC(kernel='rbf', class_weight='balanced')

    def train(self, train_data, train_label):
        temp = self._deal_data(train_data)
        # temp = self.pca.fit_transform(temp)
        # 训练SVM分类器
        self.model.fit(temp, train_label)

    def predict(self, x_test):
        temp = self._deal_data(x_test)
        return self.model.predict(temp), self.model.decision_function(temp)

    def _deal_data(self, data):
        temp = np.zeros((data.shape[0], len(self.start), data.shape[1]))
        for i in range(len(self.start)):
            temp[:, i] = data[:, :, self.start[i]: self.start[i] + self.window_len].mean(-1)
        return temp.reshape(len(temp), -1)

    def get_score(self, test_data, label):
        result, judgment = self.predict(test_data)
        auc = roc_auc_score(label, judgment)
        f1 = f1_score(label, result)
        return auc, f1, result


class BLDA:
    def __init__(self, threshold=0):
        self._fiter = None
        self._trained_items = 0
        self.scaler_list = []
        self.pca_filter = None
        data_len = 150
        self.window_len = 15
        step = int(self.window_len * 0.5)
        self.start = np.arange(0, data_len - step, step, dtype=int)
        self.threshold = threshold

    def init(self):
        self._fiter = None
        self.scaler_list = []

    def train(self, train_data, train_label):
        n_examples = len(train_label)
        n_pos_examples = np.sum(train_label == 1)

        for i in range(train_data.shape[1]):  # iterate over channels
            scaler = StandardScaler()
            train_data[:, i, :] = scaler.fit_transform(train_data[:, i, :])
            self.scaler_list.append(scaler)
        temp = self._deal_data(train_data)
        x = cp.asarray(temp.T)
        y = cp.asarray(3 - train_label * 2)
        del temp
        del train_data
        del train_label

        # compute regression targets from class labels (to do lda via regression)
        # 从类标签计算回归目标
        y[y == 1] = n_examples / n_pos_examples  # 重新给目标和非目标赋值
        y[y == -1] = - n_examples / (n_examples - n_pos_examples)

        # add feature that is constantly one (bias term)
        # 添加恒定为一的特征（偏差项）
        x = cp.vstack((x, cp.ones((1, x.shape[1]))))  # 给数据增加一行全为1

        # 变量初始化
        n_features = len(x)
        d_beta = float("inf")
        d_alpha = float("inf")
        alpha = 25
        biasalpha = 0.00000001
        beta = 1
        stop_eps = 1e-4
        i = 1
        max_it = 500
        d, v = cp.linalg.eigh(x @ x.T)  # v的正负有点问题
        vxy = v.T @ x @ y
        e = cp.ones(n_features - 1)

        # estimate alpha and beta iteratively
        # 迭代计算得到W 数据权重
        while ((d_alpha > stop_eps) or (d_beta > stop_eps)) and (i < max_it):
            alpha_old = alpha
            beta_old = beta
            m = cp.dot(beta * v, ((beta_old * d + cp.hstack((alpha * e, biasalpha))) ** -1 * vxy))
            h = y - m.T @ x
            # err = cp.sum(h * h, axis=0)
            err = h.T @ h
            h = cp.hstack((alpha * e, biasalpha))
            gamma = cp.sum((beta * d) / (beta * d + h))
            alpha = gamma / (m.T @ m)
            beta = (n_examples - gamma) / err
            d_alpha = cp.abs(alpha - alpha_old)
            d_beta = cp.abs(beta - beta_old)
            i = i + 1
            self._fiter = m

    def predict(self, data):
        # test_data = pca_white(test_data, 20)
        for i in range(data.shape[1]):  # iterate over channels
            data[:, i, :] = self.scaler_list[i].fit_transform(data[:, i, :])
        temp = self._deal_data(data)
        # temp = self.scaler.transform(test_data.reshape(test_data.shape[0], -1))
        # temp = self.pca.transform(temp)
        temp = cp.hstack((temp, cp.ones((len(temp), 1))))

        temp = cp.asarray(temp)
        return temp @ self._fiter

    def get_score(self, test_data, label):
        predict = self.predict(test_data).get()
        result = 2 - (predict > self.threshold)
        auc = roc_auc_score(3 - label * 2, predict)
        f1 = f1_score(label, result)
        return auc, f1, result

    def _deal_data(self, data):
        temp = np.zeros((data.shape[0], len(self.start), data.shape[1]))
        for i in range(len(self.start)):
            temp[:, i] = data[:, :, self.start[i]: self.start[i] + self.window_len].mean(-1)
        return temp.reshape(len(temp), -1)
