from sklearn.metrics import r2_score
import scipy.signal
import mne
import numpy as np

def process(data):
    same_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 57, 59, 60]

    ch_types = []
    data_ = []
    ch_names = []
    for i in range(len(same_index)):
        ch_types.append('eeg')
        data_.append(data[:, 0, same_index[i], 0])
        ch_names.append('Fpz')
#         ch_names.append(ch_list[same_index[i]])

    sfreq = 500.0  # 采样频率
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # 创建数据对象 data(30, 500)
    raw = mne.io.RawArray(data_, info)

    # 0.1Hz~100Hz高低通滤波，以及50Hz陷波滤波
    raw_hlfilter = raw.filter(l_freq=0.1, h_freq=100,method='fir')
    raw_hltfilter = raw_hlfilter.notch_filter(freqs=50)

    # 选取平均值为参考通道
    raw_ref = raw_hltfilter.set_eeg_reference(ref_channels = 'average')

    # 设定时间和通道
    time_start = 0.
    time_end = 1.
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    t_idx = raw.time_as_index([time_start, time_end])

    datas, times = raw_ref[picks, t_idx[0]:t_idx[1]]

    return datas

def true_data():
    same_index_raw = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    
    # 指定CNT文件路径
    cnt_file = 'D:\\TVB_Distribution\\test\\A_MI.cnt'

    # 使用mne加载CNT文件
    raw = mne.io.read_raw_cnt(cnt_file, preload=True)

    # 0.1Hz~100Hz高低通滤波，以及50Hz陷波滤波
    raw_hlfilter = raw.copy().filter(l_freq=0.1, h_freq=100,method='fir')
    raw_hltfilter = raw_hlfilter.copy().notch_filter(freqs=50)

    # 选取M1和M2为参考通道
    raw_ref = raw_hltfilter.copy().set_eeg_reference(ref_channels = ['M1', 'M2'])

    # 取出对应标签的事件信息，“1”是描述， 2是标签
    custom_mapping = {"1": 2}
    events_from_annot, event_dict = mne.events_from_annotations(raw_ref, event_id=custom_mapping)

    # 创建Epochs对象用于根据事件获得数据，tmin指事件发生前几秒，tmax指事件发生后几秒
    event_data_ref = mne.Epochs(raw_ref, events_from_annot, tmin=0.0, tmax=1.0, baseline=None, preload=True)

    # 分别获得电位差数据以及时间数据
    data = event_data_ref.get_data()
    
    # 取出第5个事件的电位差数据
    data_seg_ref = data[5]

    # f contains the frequency components
    # S is the PSD
    (f, S1)= scipy.signal.welch(data_seg_ref*1e6, 500, nperseg=1000)
    
    s = []
    for i in range(len(S1)):
        if i in same_index_raw:
            s.append(S1[i])
            
    s = np.array(s)
    
    return s

def r2_cal(data):
    
    data_true = true_data()
    print(data_true)
    
    data_pro = process(data)
    print(data_pro)
    
    (f, S)= scipy.signal.welch(data_pro, 500, nperseg=1000)
    
    r2 = r2_score(data_true[:,0:100], S[:,0:100])
    
    return r2