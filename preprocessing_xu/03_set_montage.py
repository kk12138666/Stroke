from pathlib import Path
import mne
import matplotlib.pyplot as plt

# ============================================================
# 第 3 步：设置 10-20 / 10-10 电极 montage
# 使用 standard_1020
# 并写入 MNE HTML QC 报告
# ============================================================

# ===== 1. 设置路径 =====
data_dir = Path(r"D:\EEG预处理\EEG\test03-20260617")
vhdr_path = data_dir / "huangweidong_3.vhdr"

out_dir = Path(r"D:\EEG预处理\预处理_xu\MNE_QC_huangweidong_3")
out_dir.mkdir(parents=True, exist_ok=True)

report_path = out_dir / "huangweidong_3_mne_qc_report.html"
montage_fig_path = out_dir / "montage_standard_1020.png"

# ===== 2. 读取 BrainVision 数据 =====
raw = mne.io.read_raw_brainvision(
    vhdr_path,
    preload=False,
    verbose=True
)

# ===== 3. 设置 ECG 通道类型 =====
if "ECG" in raw.ch_names:
    raw.set_channel_types({"ECG": "ecg"})
else:
    raise ValueError("没有找到 ECG 通道，请检查通道名是否叫 ECG。")

# ===== 4. 设置 standard_1020 montage =====
montage = mne.channels.make_standard_montage("standard_1020")

# on_missing="ignore" 的意思是：
# 如果某些通道名不在 standard_1020 里，不直接报错，而是继续运行
raw.set_montage(
    montage,
    on_missing="ignore"
)

# ===== 5. 检查哪些通道匹配 montage =====
montage_channel_names = set(montage.ch_names)

eeg_channels = []
ecg_channels = []
matched_eeg_channels = []
unmatched_eeg_channels = []

for ch_name, ch_type in zip(raw.ch_names, raw.get_channel_types()):
    if ch_type == "eeg":
        eeg_channels.append(ch_name)

        if ch_name in montage_channel_names:
            matched_eeg_channels.append(ch_name)
        else:
            unmatched_eeg_channels.append(ch_name)

    elif ch_type == "ecg":
        ecg_channels.append(ch_name)

print("Montage 设置完成")
print(f"EEG 通道数: {len(eeg_channels)}")
print(f"ECG 通道: {ecg_channels}")
print(f"成功匹配 montage 的 EEG 通道数: {len(matched_eeg_channels)}")
print(f"无法匹配 montage 的 EEG 通道数: {len(unmatched_eeg_channels)}")
print("无法匹配的 EEG 通道:")
print(unmatched_eeg_channels)

# ===== 6. 生成电极位置图 =====
fig = raw.plot_sensors(
    show_names=True,
    show=False
)

fig.savefig(
    montage_fig_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(f"电极位置图已保存到: {montage_fig_path}")

# ===== 7. 基础信息 =====
n_channels = len(raw.ch_names)
sfreq = raw.info["sfreq"]
n_samples = raw.n_times
duration_sec = n_samples / sfreq

channel_types = raw.get_channel_types()

type_summary = {}
for ch_type in channel_types:
    type_summary[ch_type] = type_summary.get(ch_type, 0) + 1

channel_type_rows = []
for ch_name, ch_type in zip(raw.ch_names, channel_types):
    channel_type_rows.append(
        f"<tr><td>{ch_name}</td><td>{ch_type}</td></tr>"
    )

type_summary_rows = []
for ch_type, count in type_summary.items():
    type_summary_rows.append(
        f"<tr><td>{ch_type}</td><td>{count}</td></tr>"
    )

matched_rows = []
for ch_name in matched_eeg_channels:
    matched_rows.append(f"<tr><td>{ch_name}</td></tr>")

unmatched_rows = []
for ch_name in unmatched_eeg_channels:
    unmatched_rows.append(f"<tr><td>{ch_name}</td></tr>")

ecg_rows = []
for ch_name in ecg_channels:
    ecg_rows.append(f"<tr><td>{ch_name}</td><td>ecg</td><td>Excluded from EEG montage matching</td></tr>")

# ===== 8. 重新创建 MNE Report =====
# 不使用 mne.open_report，避免 h5io 依赖
report = mne.Report(
    title="huangweidong_3 - EEG QC Report"
)

# ===== 9. 写入 Overview =====
overview_html = f"""
<h2>Overview</h2>

<table border="1" cellspacing="0" cellpadding="6">
<tr><th>Item</th><th>Value</th></tr>
<tr><td>Data directory</td><td>{data_dir}</td></tr>
<tr><td>BrainVision header</td><td>{vhdr_path.name}</td></tr>
<tr><td>Channels</td><td>{n_channels}</td></tr>
<tr><td>Sampling rate</td><td>{sfreq:.1f} Hz</td></tr>
<tr><td>Samples</td><td>{n_samples}</td></tr>
<tr><td>Duration</td><td>{duration_sec:.2f} s ({duration_sec / 60:.2f} min)</td></tr>
<tr><td>Data format</td><td>BrainVision</td></tr>
</table>

<h3>Channel names</h3>
<p>{", ".join(raw.ch_names)}</p>
"""

report.add_html(
    html=overview_html,
    title="Overview",
    section="Overview"
)

# ===== 10. 写入 Channel Types =====
channel_type_html = f"""
<h2>Channel Type Setting</h2>

<p>
本步骤将 <b>ECG</b> 通道设置为 <code>ecg</code> 类型。
其余通道保留为 <code>eeg</code> 类型。
</p>

<h3>Channel type summary</h3>

<table border="1" cellspacing="0" cellpadding="6">
<tr><th>Channel Type</th><th>Count</th></tr>
{''.join(type_summary_rows)}
</table>

<h3>All channels</h3>

<table border="1" cellspacing="0" cellpadding="6">
<tr><th>Channel Name</th><th>Channel Type</th></tr>
{''.join(channel_type_rows)}
</table>
"""

report.add_html(
    html=channel_type_html,
    title="Channel Types",
    section="Channel and Montage"
)

# ===== 11. 写入 Montage 检查结果 =====
montage_html = f"""
<h2>Montage Setting: standard_1020</h2>

<p>
本步骤使用 MNE 的 <code>standard_1020</code> montage 对 EEG 电极进行空间位置映射。
ECG 已设置为 <code>ecg</code> 类型，不参与 EEG 电极 montage 匹配。
</p>

<h3>Montage summary</h3>

<table border="1" cellspacing="0" cellpadding="6">
<tr><th>Item</th><th>Value</th></tr>
<tr><td>Montage</td><td>standard_1020</td></tr>
<tr><td>Total channels</td><td>{n_channels}</td></tr>
<tr><td>EEG channels</td><td>{len(eeg_channels)}</td></tr>
<tr><td>ECG channels</td><td>{len(ecg_channels)}</td></tr>
<tr><td>Matched EEG channels</td><td>{len(matched_eeg_channels)}</td></tr>
<tr><td>Unmatched EEG channels</td><td>{len(unmatched_eeg_channels)}</td></tr>
</table>

<h3>ECG channel handling</h3>

<table border="1" cellspacing="0" cellpadding="6">
<tr><th>Channel</th><th>Type</th><th>Note</th></tr>
{''.join(ecg_rows)}
</table>

<h3>Unmatched EEG channels</h3>
"""

if len(unmatched_eeg_channels) == 0:
    montage_html += """
<p>所有 EEG 通道均成功匹配到 standard_1020 montage。</p>
"""
else:
    montage_html += f"""
<table border="1" cellspacing="0" cellpadding="6">
<tr><th>Unmatched EEG Channel</th></tr>
{''.join(unmatched_rows)}
</table>
"""

montage_html += f"""
<h3>Matched EEG channels</h3>

<table border="1" cellspacing="0" cellpadding="6">
<tr><th>Matched EEG Channel</th></tr>
{''.join(matched_rows)}
</table>
"""

report.add_html(
    html=montage_html,
    title="Montage Summary",
    section="Channel and Montage"
)

# ===== 12. 把电极位置图加入报告 =====
report.add_image(
    image=montage_fig_path,
    title="standard_1020 Sensor Layout",
    section="Channel and Montage"
)

# ===== 13. 保存 HTML 报告 =====
report.save(
    report_path,
    overwrite=True,
    open_browser=False
)

print(f"HTML QC 报告已保存到: {report_path}")