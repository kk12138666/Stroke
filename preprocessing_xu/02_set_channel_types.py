from pathlib import Path
import mne

# ============================================================
# 第 3 步：设置通道类型
# ECG -> ecg
# 其他通道保持 eeg
# 并写入 MNE HTML QC 报告
# ============================================================

# ===== 1. 设置路径 =====
data_dir = Path(r"D:\EEG预处理\EEG\test03-20260617")
vhdr_path = data_dir / "huangweidong_3.vhdr"

out_dir = Path(r"D:\EEG预处理\预处理_xu\MNE_QC_huangweidong_3")
out_dir.mkdir(parents=True, exist_ok=True)

report_path = out_dir / "huangweidong_3_mne_qc_report.html"

# ===== 2. 读取 BrainVision 数据 =====
# MNE 会根据 .vhdr 自动读取同目录下的 .vmrk 和 .eeg
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

# ===== 4. 获取基础信息 =====
n_channels = len(raw.ch_names)
sfreq = raw.info["sfreq"]
n_samples = raw.n_times
duration_sec = n_samples / sfreq

channel_types = raw.get_channel_types()

type_summary = {}
for ch_type in channel_types:
    type_summary[ch_type] = type_summary.get(ch_type, 0) + 1

print("通道类型设置完成")
print(type_summary)

# ===== 5. 准备通道类型表格 =====
channel_type_rows = []

for ch_name, ch_type in zip(raw.ch_names, channel_types):
    channel_type_rows.append(
        f"<tr><td>{ch_name}</td><td>{ch_type}</td></tr>"
    )

channel_type_table = "\n".join(channel_type_rows)

type_summary_rows = []
for ch_type, count in type_summary.items():
    type_summary_rows.append(
        f"<tr><td>{ch_type}</td><td>{count}</td></tr>"
    )

type_summary_table = "\n".join(type_summary_rows)

# ===== 6. 创建新的 MNE Report =====
# 注意：这里不使用 mne.open_report，所以不需要 h5io
# 每一步都重新生成完整 HTML，避免依赖问题
report = mne.Report(
    title="huangweidong_3 - EEG QC Report"
)

# ===== 7. 写入第 1 步 Overview 内容 =====
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

# ===== 8. 写入第 2 步 Channel Types 内容 =====
channel_type_html = f"""
<h2>Channel Type Setting</h2>

<p>
本步骤将 <b>ECG</b> 通道设置为 <code>ecg</code> 类型。
其余通道保留为 <code>eeg</code> 类型。
</p>

<p>
这样做的原因是：ECG 是心电参考/生理通道，振幅和频谱特征与普通 EEG 不同。
如果不单独设置，后续坏导检测、PSD 检查、epoch rejection 时可能会把 ECG 误判为坏 EEG 通道。
</p>

<h3>Channel type summary</h3>

<table border="1" cellspacing="0" cellpadding="6">
<tr><th>Channel Type</th><th>Count</th></tr>
{type_summary_table}
</table>

<h3>All channels</h3>

<table border="1" cellspacing="0" cellpadding="6">
<tr><th>Channel Name</th><th>Channel Type</th></tr>
{channel_type_table}
</table>
"""

report.add_html(
    html=channel_type_html,
    title="Channel Types",
    section="Channel and Montage"
)

# ===== 9. 保存 HTML 报告 =====
report.save(
    report_path,
    overwrite=True,
    open_browser=False
)

print(f"HTML QC 报告已保存到: {report_path}")