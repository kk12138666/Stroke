from pathlib import Path
import mne

# ===== 1. 设置数据路径 =====
data_dir = Path(r"D:\EEG预处理\EEG\test03-20260617")
vhdr_path = data_dir / "huangweidong_3.vhdr"

# ===== 2. 设置报告输出路径 =====
out_dir = Path(r"D:\EEG预处理\预处理_xu\MNE_QC_huangweidong_3")
out_dir.mkdir(parents=True, exist_ok=True)
report_path = out_dir / "huangweidong_3_mne_qc_report.html"

# ===== 3. 用 MNE 读取 BrainVision 数据 =====
# MNE 会通过 .vhdr 自动找到同目录下的 .vmrk 和 .eeg
raw = mne.io.read_raw_brainvision(
    vhdr_path,
    preload=False,
    verbose=True
)

# ===== 4. 提取基础信息 =====
n_channels = len(raw.ch_names)
sfreq = raw.info["sfreq"]
n_samples = raw.n_times
duration_sec = n_samples / sfreq

print("读取成功")
print(f"文件: {vhdr_path}")
print(f"通道数: {n_channels}")
print(f"采样率: {sfreq} Hz")
print(f"采样点数: {n_samples}")
print(f"数据时长: {duration_sec:.2f} 秒 / {duration_sec / 60:.2f} 分钟")
print("通道名:")
print(raw.ch_names)

# ===== 5. 创建 MNE HTML Report =====
report = mne.Report(
    title="huangweidong_3 - EEG QC Report"
)

overview_html = f"""
<h2>Overview</h2>

<table>
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

# ===== 6. 保存 HTML 报告 =====
report.save(
    report_path,
    overwrite=True,
    open_browser=False
)

print(f"HTML QC 报告已保存到: {report_path}")