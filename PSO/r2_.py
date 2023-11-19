from sklearn.metrics import r2_score
import scipy.signal
import mne
import numpy as np

def process(data):
    same_index = [23, 13, 43, 0, 29, 37, 2, 14, 57, 18, 10, 31, 44, 24, 21, 41, 4, 50, 15, 9, 26, 30, 11, 38, 32, 56, 60, 19, 22, 25, 36, 61, 12, 35, 49, 20, 40, 51, 52, 39, 48, 16, 5, 17, 7, 3, 47, 46, 42, 6, 45, 1, 8, 55, 59]

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
    picks = mne.pick_types(raw_ref.info, eeg=True, exclude='bads')
    t_idx = raw_ref.time_as_index([time_start, time_end])

    datas, times = raw_ref[picks, t_idx[0]:t_idx[1]]

    (f, S) = scipy.signal.welch(datas, 500, nperseg=500)

    return S

def true_data():
    same_index_raw = [21, 33, 5, 1, 8, 19, 10, 47, 0, 42, 14, 26, 6, 22, 34, 50, 28, 46, 48, 58, 39, 9, 15, 20, 27, 41, 56, 17, 35, 38, 4, 16, 32, 3, 45, 18, 37, 54, 55, 36, 31, 7, 29, 25, 44, 11, 30, 13, 51, 43, 12, 2, 57, 40, 49]

    # 指定CNT文件路径
    cnt_file = '/public/home/ynhang/yuze/code/multitask/data/eyeclosed.fif'

    # 使用mne加载CNT文件
    raw = mne.io.read_raw_fif(cnt_file, preload=True)

    # 设置新的采样率（以毫秒为单位）
    new_sfreq = 500  
    raw.resample(new_sfreq)

    # 使用mne加载CNT文件
    time_start = 2.0
    time_end = 3.0
    # raw = mne.io.read_raw_cnt(cnt_file, preload=True)
    raw = mne.io.read_raw_fif(cnt_file, preload=True)
    # 0.1Hz~100Hz高低通滤波，以及50Hz陷波滤波
    raw_hlfilter = raw.copy().filter(l_freq=0.1, h_freq=100,method='fir')
    raw_hltfilter = raw_hlfilter.copy().notch_filter(freqs=50)

    # 选取M1和M2为参考通道
    raw_ref = raw_hltfilter.copy().set_eeg_reference(ref_channels = 'average')

    # 取出对应标签的事件信息，“1”是描述， 2是标签
    # custom_mapping = {"1": 2}
    # events_from_annot, event_dict = mne.events_from_annotations(raw_ref, event_id=custom_mapping)

    # 创建Epochs对象用于根据事件获得数据，tmin指事件发生前几秒，tmax指事件发生后几秒
    # event_data_ref = mne.Epochs(raw_ref, events_from_annot, tmin=0.0, tmax=1.0, baseline=None, preload=True)

    # 分别获得电位差数据以及时间数据
    # data = event_data_ref.get_data()
    
    # 取出第5个事件的电位差数据
    # data_seg_ref = data[5]
    

    # 设定时间和通道
    picks = mne.pick_types(raw_ref.info, eeg=True, exclude='bads')
    t_idx = raw_ref.time_as_index([time_start, time_end])
    data, times = raw_ref[picks, t_idx[0]:t_idx[1]]
    # f contains the frequency components
    # S is the PSD
    (f, S1)= scipy.signal.welch(data*1e6, 500, nperseg=500)
    
    s = []
    for i in range(len(same_index_raw)):
        s.append((S1[same_index_raw[i]]))
            
    s = np.array(s)
    
    return s

def r2_cal(data):
    temp = open("r2.txt", "w")
    temp.write("r2_score begin\n")
    data_true = true_data()
    
    data_pro = process(data)
    
    r2 = r2_score(data_true[:,1:100], (data_pro[:,1:100]))
    
    sum = 0
    for i in range(len(data_true)):
        t = r2_score((data_true[i,1:100]), (data_pro[i, 1:100]))
        sum = sum + t
        temp.write(f"{i}channl: {t}\n")

    sum1 = sum / len(data_true)
    temp.write(f"sum: {sum1}, sum/{len(data_true)}: {sum} r2: {r2}\n")
    temp.write("r2_score over\n")
    temp.close()
    
    return sum1
