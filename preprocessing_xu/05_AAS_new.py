from pathlib import Path
from tools import read_data_bp, get_timetree, reshape_eegdata
import numpy as np
import mne
import matplotlib.pyplot as plt

# ============================================================
# 1. 路径设置
# ============================================================
VHDR_PATH = Path(r"D:\EEG预处理\EEG\test03-20260617\huangweidong_3.vhdr")
VMRK_PATH = Path(r"D:\EEG预处理\EEG\test03-20260617\huangweidong_3.vmrk")

OUT_DIR = VHDR_PATH.parent / "pure_eeg_aas_fp1_no_interp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# OBS-PCA 下一步要读取这个文件
OUT_NPZ = OUT_DIR / f"{VHDR_PATH.stem}_aas_epochs_for_obs_pca.npz"

# ============================================================
# 2. AAS 参数
# ============================================================
ARTIFACT_EVENT_KEYWORDS = [
    "Stimulus/S  7"
]

TMIN = -0.02
TMAX = 0.08

LOCAL_EVENT_WINDOW = 2

EEG_ONLY = True

FILTER_FOR_TEMPLATE = False
L_FREQ = 0.1
H_FREQ = 100.0


# ============================================================
# 3. 结果图设置
# ============================================================
CHECK_CHANNEL_NAME = "Fp1"

PLOT_START_SAMPLE = 5000 * 10
PLOT_N_SAMPLES = 5000 * 0.05

PLOT_IN_UV = True


# ============================================================
# 4. 工具函数
# ============================================================

def find_artifact_events(raw, keywords):
    """
    从 raw.annotations 中找用于 AAS 的事件 onset，单位是秒。
    """
    if raw.annotations is None or len(raw.annotations) == 0:
        raise RuntimeError("当前 raw 中没有 annotations/marker，无法做事件锁定 AAS。")

    print("\n当前数据中的所有 annotations:")
    for desc in sorted(set(raw.annotations.description)):
        print("  ", desc)

    selected_onsets = []
    selected_descs = []

    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        desc_str = str(desc)
        if any(key.lower() in desc_str.lower() for key in keywords):
            selected_onsets.append(onset)
            selected_descs.append(desc_str)

    selected_onsets = np.array(selected_onsets, dtype=float)

    if len(selected_onsets) == 0:
        raise RuntimeError(
            "没有找到匹配 ARTIFACT_EVENT_KEYWORDS 的事件。\n"
            "请先查看上面打印的 annotation 名称，然后修改 ARTIFACT_EVENT_KEYWORDS。"
        )

    print("\n本次用于 AAS 的 annotations:")
    for desc in sorted(set(selected_descs)):
        print("  ", desc)

    print(f"本次用于 AAS 的事件数量: {len(selected_onsets)}")

    return selected_onsets


def extract_epochs_no_interp(x, sfreq, event_onsets_sec, tmin, tmax):
    """
    不插值，直接把事件 onset 四舍五入到最近整数采样点，然后截取 epoch。
    """
    n_channels, n_samples = x.shape

    pre_samples = int(round(abs(tmin) * sfreq))
    post_samples = int(round(tmax * sfreq))
    rel = np.arange(-pre_samples, post_samples + 1)

    event_samples = np.round(event_onsets_sec * sfreq).astype(int)

    valid_mask = (
        (event_samples - pre_samples >= 0)
        & (event_samples + post_samples < n_samples)
    )

    valid_event_samples = event_samples[valid_mask]

    if len(valid_event_samples) == 0:
        raise RuntimeError("没有有效事件可用于截取 epoch，请检查 TMIN/TMAX 或 marker 时间。")

    epochs = np.stack([
        x[:, event_sample - pre_samples:event_sample + post_samples + 1]
        for event_sample in valid_event_samples
    ])

    return epochs, valid_event_samples, valid_mask, rel


def make_aas_templates(epochs, local_event_window=None):
    """
    Windowed-AAS：为每个事件生成局部平均模板。
    输入/输出 shape: [n_events, n_channels, epoch_len]
    """
    n_events = epochs.shape[0]
    templates = np.zeros_like(epochs)

    if local_event_window is None:
        global_template = epochs.mean(axis=0, keepdims=True)
        templates[:] = global_template
        return templates

    for i in range(n_events):
        lo = max(0, i - local_event_window)
        hi = min(n_events, i + local_event_window + 1)

        use_idx = np.arange(lo, hi)
        use_idx = use_idx[use_idx != i]

        if len(use_idx) == 0:
            templates[i] = 0
        else:
            templates[i] = epochs[use_idx].mean(axis=0)

    return templates


def subtract_templates_no_interp(raw_data, event_samples, templates, rel):
    """
    不插值，直接把每个 AAS 模板放回整数采样点位置，再从原始 EEG 中扣除。
    这个函数用于连续 raw 数据版本；当前主流程保存 epoch 结果，所以暂时不调用。
    """
    artifact_sum = np.zeros_like(raw_data)
    weight = np.zeros_like(raw_data)

    _, n_samples = raw_data.shape

    for i, event_sample in enumerate(event_samples):
        idx = event_sample + rel

        valid = (idx >= 0) & (idx < n_samples)
        valid_idx = idx[valid]

        artifact_sum[:, valid_idx] += templates[i][:, valid]
        weight[:, valid_idx] += 1

    artifact = np.zeros_like(raw_data)
    mask = weight > 0
    artifact[mask] = artifact_sum[mask] / weight[mask]

    cleaned = raw_data - artifact

    return cleaned, artifact


def Parser_Event_Related_EEG(
    eeg_hdr: str,
    eeg_mrk: str,
    samplerate: int,
    duration: int,
    savepath: str = None,
    write: bool = False
):
    """
    读取 BrainVision 数据，并按 marker 切成事件相关 EEG 数据。
    返回通常为 shape = [n_channels, epoch_len, n_events]
    """
    raw_eeg_data, _ = read_data_bp(eeg_hdr)

    timetree = get_timetree(eeg_mrk)

    parser_eeg = reshape_eegdata(raw_eeg_data, samplerate, duration, timetree)

    return parser_eeg


def plot_fp1_raw_and_aas(raw_data, cleaned_data, ch_idx, sfreq):
    """
    画 Fp1 原始 epoch 和 Windowed-AAS 后 epoch。
    raw_data / cleaned_data shape = [n_events, n_channels, epoch_len]
    """
    n_samples = raw_data.shape[-1]
    start = max(0, int(PLOT_START_SAMPLE))
    stop = min(start + int(PLOT_N_SAMPLES), n_samples)

    if stop <= start:
        raise RuntimeError("绘图采样点范围无效，请检查 PLOT_START_SAMPLE 和 PLOT_N_SAMPLES。")

    x_axis = np.arange(stop - start)

    # 这里画第 4 个 epoch；如果想换，可以改成 0、1、2...
    plot_epoch_idx = 3

    if plot_epoch_idx >= raw_data.shape[0]:
        raise RuntimeError(
            f"plot_epoch_idx={plot_epoch_idx} 超出 epoch 数量范围，"
            f"当前共有 {raw_data.shape[0]} 个 epoch。"
        )

    raw_y = raw_data[plot_epoch_idx, ch_idx, start:stop]
    cleaned_y = cleaned_data[plot_epoch_idx, ch_idx, start:stop]

    if PLOT_IN_UV:
        raw_y = raw_y * 1e6
        cleaned_y = cleaned_y * 1e6
        ylabel = "uV"
    else:
        ylabel = "V"

    print(f"[DEBUG]: np.sum(raw_y-cleaned_y): {np.sum(raw_y - cleaned_y)}")

    fig, axs = plt.subplots(
        3,
        1,
        figsize=(14, 4),
        sharex=True,
        constrained_layout=True
    )

    axs[0].plot(x_axis, raw_y, color="blue", linewidth=0.8)
    axs[0].set_title("Raw Data")
    axs[0].set_ylabel(ylabel)

    axs[1].plot(x_axis, cleaned_y, color="orange", linewidth=0.8)
    axs[1].set_title("Windowed-AAS")
    axs[1].set_ylabel(ylabel)

    axs[2].plot(x_axis, raw_y - cleaned_y, color="red", linewidth=0.8)
    axs[2].set_title("Difference")
    axs[2].set_ylabel(ylabel)
    axs[2].set_xlabel("Sample")

    plt.show()


# ============================================================
# 5. 主流程
# ============================================================

def main():
    print(f"读取数据: {VHDR_PATH}")
    raw = mne.io.read_raw_brainvision(VHDR_PATH, preload=True)

    print(raw)
    print(f"采样率: {raw.info['sfreq']} Hz")
    print(f"总通道数: {len(raw.ch_names)}")
    print(f"数据时长: {raw.times[-1]:.2f} 秒")

    if EEG_ONLY:
        picks = mne.pick_types(
            raw.info,
            eeg=True,
            eog=False,
            ecg=False,
            emg=False,
            stim=False,
            exclude=[],
        )
    else:
        picks = np.arange(len(raw.ch_names))

    if len(picks) == 0:
        raise RuntimeError("没有找到 EEG 通道。")

    ch_names = [raw.ch_names[p] for p in picks]

    print(f"参与 AAS 的 EEG 通道数: {len(picks)}")
    print(ch_names)

    if CHECK_CHANNEL_NAME not in ch_names:
        raise RuntimeError(
            f"检查图通道 {CHECK_CHANNEL_NAME} 不在 EEG 通道中。\n"
            f"可用通道包括: {ch_names}"
        )

    find_artifact_events(raw, ARTIFACT_EVENT_KEYWORDS)

    # --------------------------------------------------------
    # 读取并切成 20 秒任务 epoch
    # x 通常 shape = [n_channels, epoch_len, n_events]
    # --------------------------------------------------------
    x = Parser_Event_Related_EEG(
        eeg_hdr=VHDR_PATH,
        eeg_mrk=VMRK_PATH,
        samplerate=int(raw.info["sfreq"]),
        duration=20,
        write=False
    )

    print(f"原始事件相关 EEG shape: {x.shape}")

    # --------------------------------------------------------
    # OBS-PCA 需要 shape = [n_events, n_channels, epoch_len]
    # 如果 x = [n_channels, epoch_len, n_events]，
    # 那么 transpose 后就是 [n_events, n_channels, epoch_len]
    # --------------------------------------------------------
    i, j, k = x.shape
    epochs = x.reshape(k, i, j)

    print(f"事件 epoch shape: {epochs.shape}")

    # --------------------------------------------------------
    # Windowed-AAS
    # --------------------------------------------------------
    templates = make_aas_templates(
        epochs,
        local_event_window=LOCAL_EVENT_WINDOW,
    )

    cleaned_epochs = epochs - templates

    print(f"AAS templates shape: {templates.shape}")
    print(f"cleaned_epochs shape: {cleaned_epochs.shape}")
    print(
        f"[DEBUG]: np.sum(epochs-cleaned_epochs): "
        f"{np.sum(epochs[0, 0, :] - cleaned_epochs[0, 0, :])}"
    )

    # --------------------------------------------------------
    # 保存给 OBS-PCA 使用的三维 AAS 后 epoch 数据
    # --------------------------------------------------------
    np.savez_compressed(
        OUT_NPZ,
        aas_epochs=cleaned_epochs.astype(np.float32),
        raw_epochs=epochs.astype(np.float32),
        aas_templates=templates.astype(np.float32),
        ch_names=np.array(ch_names),
        sfreq=float(raw.info["sfreq"]),
        local_event_window=LOCAL_EVENT_WINDOW,
        epoch_shape=np.array(cleaned_epochs.shape),
    )

    print(f"已保存 OBS-PCA 输入文件: {OUT_NPZ}")
    print(f"OBS-PCA 输入 aas_epochs shape: {cleaned_epochs.shape}")

    # --------------------------------------------------------
    # 只画 Fp1 检查图
    # --------------------------------------------------------
    fp1_idx = ch_names.index(CHECK_CHANNEL_NAME)

    print(f"检查通道: {CHECK_CHANNEL_NAME} 在索引: {fp1_idx}")

    plot_fp1_raw_and_aas(
        raw_data=epochs,
        cleaned_data=cleaned_epochs,
        ch_idx=fp1_idx,
        sfreq=raw.info["sfreq"],
    )

    print("\n完成。")


if __name__ == "__main__":
    main()