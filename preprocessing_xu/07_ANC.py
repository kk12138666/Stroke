from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 路径设置
# ============================================================
OBS_NPZ_PATH = Path(
    r"D:\EEG预处理\EEG\test03-20260617\pure_eeg_aas_fp1_no_interp\obs_pca_after_aas\huangweidong_3_obs_pca_after_aas_epochs.npz"
)

OUT_DIR = OBS_NPZ_PATH.parent / "anc_after_obs_pca"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_NPZ = OUT_DIR / "huangweidong_3_anc_after_obs_pca_epochs.npz"


# ============================================================
# 2. ANC 参数
# ============================================================

# 参考电极通道

REF_CHANNEL_NAMES = ["ECG"]

# 每次回归处理多少采样点
# 5000 Hz 下，20000 点 = 4 秒
ANC_BLOCK = 20000

# ridge 正则化强度，防止矩阵求解不稳定
ANC_RIDGE = 1e-3


# ============================================================
# 3. 绘图参数
# ============================================================
CHECK_CHANNEL_NAME = "Fp1"

PLOT_EPOCH_INDEX = 3
PLOT_START_SAMPLE = 5000 * 10
PLOT_N_SAMPLES = int(5000 * 0.05)

PLOT_IN_UV = True


# ============================================================
# 4. ANC 函数
# ============================================================

def ANC_Regression(X_eeg, refs, block=20000, ridge=1e-3):
    """
    使用参考电极通道进行自适应噪声抑制。

    Parameters
    ----------
    X_eeg : np.ndarray
        shape = [n_eeg_channels, n_samples]
        要被清理的 EEG 数据。
    refs : np.ndarray
        shape = [n_ref_channels, n_samples]
        参考电极通道，例如 EOG / ECG / EMG。
    block : int
        分块长度。
    ridge : float
        岭回归正则项。

    Returns
    -------
    Y : np.ndarray
        shape = [n_eeg_channels, n_samples]
        ANC 清理后的 EEG 数据。
    """
    Y = X_eeg.copy()
    n_samples = X_eeg.shape[-1]

    for lo in range(0, n_samples, block):
        hi = min(n_samples, lo + block)

        R = refs[:, lo:hi].T
        R = np.column_stack([R, np.ones(hi - lo)])

        Xb = X_eeg[:, lo:hi].T

        beta = np.linalg.solve(
            R.T @ R + ridge * np.eye(R.shape[1]),
            R.T @ Xb,
        )

        noise = R @ beta
        Y[:, lo:hi] = (Xb - noise).T

    return Y


def apply_anc_to_epochs(epochs, ch_names, ref_channel_names, block=20000, ridge=1e-3):
    """
    对 OBS-PCA 后的 epoch 数据逐个 epoch 做 ANC。

    Parameters
    ----------
    epochs : np.ndarray
        shape = [n_events, n_channels, epoch_len]
    ch_names : list[str]
        通道名。
    ref_channel_names : list[str]
        参考通道名。

    Returns
    -------
    anc_epochs : np.ndarray
        shape = [n_events, n_channels, epoch_len]
        ANC 后的数据。
    """
    ch_names = [str(ch) for ch in ch_names]

    missing_refs = [ch for ch in ref_channel_names if ch not in ch_names]
    if missing_refs:
        raise RuntimeError(
            f"参考通道不存在: {missing_refs}\n"
            f"当前可用通道: {ch_names}"
        )

    ref_idx = np.array([ch_names.index(ch) for ch in ref_channel_names])
    target_idx = np.array([i for i in range(len(ch_names)) if i not in ref_idx])

    print(f"ANC 参考通道: {ref_channel_names}")
    print(f"参考通道索引: {ref_idx.tolist()}")
    print(f"被清理通道数量: {len(target_idx)}")

    anc_epochs = epochs.copy()

    for ep in range(epochs.shape[0]):
        print(f"正在处理 epoch {ep + 1}/{epochs.shape[0]}")

        refs = epochs[ep, ref_idx, :]
        X_eeg = epochs[ep, target_idx, :]

        cleaned_targets = ANC_Regression(
            X_eeg,
            refs,
            block=block,
            ridge=ridge,
        )

        anc_epochs[ep, target_idx, :] = cleaned_targets

        # 参考通道本身保持原样，不做回归扣除
        anc_epochs[ep, ref_idx, :] = epochs[ep, ref_idx, :]

    return anc_epochs


# ============================================================
# 5. 绘图函数
# ============================================================

def plot_anc_result(before_epochs, after_epochs, ch_names):
    ch_names = [str(ch) for ch in ch_names]

    if CHECK_CHANNEL_NAME not in ch_names:
        raise RuntimeError(f"{CHECK_CHANNEL_NAME} 不在通道列表中。")

    ch_idx = ch_names.index(CHECK_CHANNEL_NAME)

    if PLOT_EPOCH_INDEX < 0 or PLOT_EPOCH_INDEX >= before_epochs.shape[0]:
        raise RuntimeError(
            f"PLOT_EPOCH_INDEX={PLOT_EPOCH_INDEX} 超出范围，"
            f"当前 epoch 数量为 {before_epochs.shape[0]}"
        )

    start = int(PLOT_START_SAMPLE)
    stop = min(start + int(PLOT_N_SAMPLES), before_epochs.shape[-1])

    if stop <= start:
        raise RuntimeError("绘图采样点范围无效，请检查 PLOT_START_SAMPLE 和 PLOT_N_SAMPLES。")

    x_axis = np.arange(stop - start)

    y_before = before_epochs[PLOT_EPOCH_INDEX, ch_idx, start:stop]
    y_after = after_epochs[PLOT_EPOCH_INDEX, ch_idx, start:stop]
    y_diff = y_before - y_after

    if PLOT_IN_UV:
        y_before = y_before * 1e6
        y_after = y_after * 1e6
        y_diff = y_diff * 1e6
        ylabel = "uV"
    else:
        ylabel = "V"

    fig, axs = plt.subplots(
        3,
        1,
        figsize=(14, 5),
        sharex=True,
        constrained_layout=True,
    )

    axs[0].plot(x_axis, y_before, color="green", linewidth=0.8)
    axs[0].set_title("AAS + OBS-PCA Cleaned Epoch")
    axs[0].set_ylabel(ylabel)


    axs[1].plot(x_axis, y_diff, color="red", linewidth=0.8)
    axs[1].set_title("ANC Removed Component")
    axs[1].set_ylabel(ylabel)
    
    axs[2].plot(x_axis, y_after, color="purple", linewidth=0.8)
    axs[2].set_title("After ANC")
    axs[2].set_ylabel(ylabel)
    axs[2].set_xlabel("Sample")

    plt.show()


# ============================================================
# 6. 主流程
# ============================================================

def main():
    print(f"读取 OBS-PCA 结果: {OBS_NPZ_PATH}")

    data = np.load(OBS_NPZ_PATH, allow_pickle=True)

    # 优先读取 OBS-PCA 后的数据
    if "obs_cleaned_epochs" in data:
        obs_cleaned_epochs = data["obs_cleaned_epochs"]
    elif "cleaned_epochs" in data:
        obs_cleaned_epochs = data["cleaned_epochs"]
    else:
        raise RuntimeError(
            "NPZ 中没有找到 obs_cleaned_epochs 或 cleaned_epochs。"
        )

    ch_names = data["ch_names"]
    sfreq = float(data["sfreq"])

    print(f"OBS-PCA 后 epochs shape: {obs_cleaned_epochs.shape}")
    print(f"采样率: {sfreq}")
    print(f"通道数: {len(ch_names)}")

    anc_cleaned_epochs = apply_anc_to_epochs(
        obs_cleaned_epochs,
        ch_names=ch_names,
        ref_channel_names=REF_CHANNEL_NAMES,
        block=ANC_BLOCK,
        ridge=ANC_RIDGE,
    )

    print(f"ANC 后 epochs shape: {anc_cleaned_epochs.shape}")

    np.savez_compressed(
        OUT_NPZ,
        anc_cleaned_epochs=anc_cleaned_epochs.astype(np.float32),
        obs_cleaned_epochs=obs_cleaned_epochs.astype(np.float32),
        ch_names=ch_names,
        sfreq=sfreq,
        ref_channel_names=np.array(REF_CHANNEL_NAMES),
        anc_block=ANC_BLOCK,
        anc_ridge=ANC_RIDGE,
    )

    print(f"已保存 ANC 结果: {OUT_NPZ}")

    plot_anc_result(
        before_epochs=obs_cleaned_epochs,
        after_epochs=anc_cleaned_epochs,
        ch_names=ch_names,
    )

    print("\n完成。")


if __name__ == "__main__":
    main()