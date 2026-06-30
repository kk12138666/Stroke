from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 路径设置
# ============================================================
AAS_NPZ_PATH = Path(
    r"D:\EEG预处理\EEG\test03-20260617\pure_eeg_aas_fp1_no_interp\huangweidong_3_aas_epochs_for_obs_pca.npz"
)

OUT_DIR = AAS_NPZ_PATH.parent / "obs_pca_after_aas"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_NPZ = OUT_DIR / "huangweidong_3_obs_pca_after_aas_epochs.npz"


# ============================================================
# 2. OBS-PCA 参数
# ============================================================
N_COMPONENT = 3

CHECK_CHANNEL_NAME = "Fp1"

PLOT_EPOCH_INDEX = 3
PLOT_START_SAMPLE = 5000 * 10
PLOT_N_SAMPLES = 5000 * 0.05

PLOT_IN_UV = True


# ============================================================
# 3. OBS-PCA 函数
# ============================================================

def Obs_Pca_Epochs(AAS_Epochs: np.ndarray, mode: str, n_component: int = 3):
    n_events, n_channels, epoch_len = AAS_Epochs.shape

    artifact_epochs = np.zeros_like(AAS_Epochs)
    cleaned_epochs = np.zeros_like(AAS_Epochs)

    for ch in range(n_channels):
        R = AAS_Epochs[:, ch, :]
        R_centered = R - R.mean(axis=0, keepdims=True)

        U, S, Vt = np.linalg.svd(R_centered, full_matrices=False)
        P = Vt[:n_component].T

        for i in range(n_events):
            r = R[i]
            artifact_epochs[i, ch, :] = P @ (P.T @ r)
            cleaned_epochs[i, ch, :] = r - P @ (P.T @ r)

    if mode == "A":
        return artifact_epochs
    else:
        return cleaned_epochs


# ============================================================
# 4. 绘图函数
# ============================================================

def plot_obs_result(aas_epochs, obs_artifacts, obs_cleaned, ch_names, sfreq):
    ch_names = [str(ch) for ch in ch_names]

    if CHECK_CHANNEL_NAME not in ch_names:
        raise RuntimeError(f"{CHECK_CHANNEL_NAME} 不在通道列表中。")

    ch_idx = ch_names.index(CHECK_CHANNEL_NAME)

    if PLOT_EPOCH_INDEX < 0 or PLOT_EPOCH_INDEX >= aas_epochs.shape[0]:
        raise RuntimeError(
            f"PLOT_EPOCH_INDEX={PLOT_EPOCH_INDEX} 超出范围，"
            f"当前 epoch 数量为 {aas_epochs.shape[0]}"
        )

    start = int(PLOT_START_SAMPLE)
    stop = min(start + int(PLOT_N_SAMPLES), aas_epochs.shape[-1])

    x_axis = np.arange(stop - start)

    y_aas = aas_epochs[PLOT_EPOCH_INDEX, ch_idx, start:stop]
    y_artifact = obs_artifacts[PLOT_EPOCH_INDEX, ch_idx, start:stop]
    y_cleaned = obs_cleaned[PLOT_EPOCH_INDEX, ch_idx, start:stop]

    if PLOT_IN_UV:
        y_aas = y_aas * 1e6
        y_artifact = y_artifact * 1e6
        y_cleaned = y_cleaned * 1e6
        ylabel = "uV"
    else:
        ylabel = "V"

    fig, axs = plt.subplots(
        3,
        1,
        figsize=(14, 5),
        sharex=True,
        constrained_layout=True
    )

    axs[0].plot(x_axis, y_aas, color="orange", linewidth=0.8)
    axs[0].set_title("AAS Cleaned Epoch")
    axs[0].set_ylabel(ylabel)

    axs[1].plot(x_axis, y_artifact, color="red", linewidth=0.8)
    axs[1].set_title("OBS-PCA Artifact")
    axs[1].set_ylabel(ylabel)

    axs[2].plot(x_axis, y_cleaned, color="green", linewidth=0.8)
    axs[2].set_title("AAS + OBS-PCA Cleaned Epoch")
    axs[2].set_ylabel(ylabel)
    axs[2].set_xlabel("Sample")

    plt.show()


# ============================================================
# 5. 主流程
# ============================================================

def main():
    print(f"读取 AAS 结果: {AAS_NPZ_PATH}")

    data = np.load(AAS_NPZ_PATH, allow_pickle=True)

    AAS_Epochs = data["aas_epochs"]
    raw_epochs = data["raw_epochs"]
    aas_templates = data["aas_templates"]
    ch_names = data["ch_names"]
    sfreq = float(data["sfreq"])

    print(f"AAS_Epochs shape: {AAS_Epochs.shape}")
    print(f"raw_epochs shape: {raw_epochs.shape}")
    print(f"aas_templates shape: {aas_templates.shape}")
    print(f"采样率: {sfreq}")
    print(f"通道数: {len(ch_names)}")

    artifact_epochs = Obs_Pca_Epochs(
        AAS_Epochs,
        mode="A",
        n_component=N_COMPONENT,
    )

    cleaned_epochs = Obs_Pca_Epochs(
        AAS_Epochs,
        mode="C",
        n_component=N_COMPONENT,
    )

    print(f"OBS-PCA artifact_epochs shape: {artifact_epochs.shape}")
    print(f"OBS-PCA cleaned_epochs shape: {cleaned_epochs.shape}")

    np.savez_compressed(
        OUT_NPZ,
        obs_cleaned_epochs=cleaned_epochs.astype(np.float32),
        obs_artifact_epochs=artifact_epochs.astype(np.float32),
        aas_epochs=AAS_Epochs.astype(np.float32),
        raw_epochs=raw_epochs.astype(np.float32),
        aas_templates=aas_templates.astype(np.float32),
        ch_names=ch_names,
        sfreq=sfreq,
        n_component=N_COMPONENT,
    )

    print(f"已保存 OBS-PCA 结果: {OUT_NPZ}")

    plot_obs_result(
        aas_epochs=AAS_Epochs,
        obs_artifacts=artifact_epochs,
        obs_cleaned=cleaned_epochs,
        ch_names=ch_names,
        sfreq=sfreq,
    )

    print("\n完成。")


if __name__ == "__main__":
    main()