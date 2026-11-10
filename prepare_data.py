"""GIAI ĐOẠN 1a: lấy mẫu khung hình từ video camera cố định, tạo ảnh nền, chia dev/test theo THỜI GIAN.
Chạy: python prepare_data.py [--video data/raw/vtest.avi] [--step 15] [--dev_ratio 0.5]
"""
import argparse, os, cv2, numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--video", default="data/raw/vtest.avi")
ap.add_argument("--step", type=int, default=15, help="lấy 1 khung mỗi N khung")
ap.add_argument("--bg_step", type=int, default=8, help="lấy 1 khung mỗi N khung để ước lượng nền")
ap.add_argument("--dev_ratio", type=float, default=0.5)
ap.add_argument("--out", default="data")
a = ap.parse_args()

# Refuse before writing frames/background, not after the writes completed.
if os.path.exists(a.out) and os.listdir(a.out):
    raise SystemExit("Thư mục đầu ra đã có dữ liệu; không ghi đè. Dùng --out thư mục mới hoặc prepare_week1.py.")
os.makedirs(a.out, exist_ok=True)

cap = cv2.VideoCapture(a.video)
n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
samples, bg_frames, i = {}, [], 0
while True:
    ok, f = cap.read()
    if not ok: break
    if i % a.step == 0: samples[i] = f
    if i % a.bg_step == 0: bg_frames.append(f)
    i += 1

# Ảnh nền = trung vị theo thời gian (người đi qua mỗi điểm ảnh chỉ trong thời gian ngắn -> trung vị loại bỏ họ).
# Thiết lập ngoại tuyến: dùng khung tương lai dù không dùng nhãn; không phải nền causal/online.
bg = np.median(np.stack(bg_frames), axis=0).astype(np.uint8)
cv2.imwrite(f"{a.out}/background.png", bg)

idx = sorted(samples)
n_dev = int(len(idx) * a.dev_ratio)
rows = []
for k, fi in enumerate(idx):
    split = "dev" if k < n_dev else "test"          # nửa đầu video = dev, nửa sau = test
    os.makedirs(f"{a.out}/frames/{split}", exist_ok=True)
    name = f"{split}/f{fi:04d}.png"
    cv2.imwrite(f"{a.out}/frames/{name}", samples[fi])
    rows.append((name, "", split))
gt_path = f"{a.out}/ground_truth.csv"
if os.path.exists(gt_path):
    print(f"[!] {gt_path} đã tồn tại, KHÔNG ghi đè (tránh mất nhãn). Xóa tay nếu muốn tạo lại.")
else:
    pd.DataFrame(rows, columns=["image", "count", "split"]).to_csv(gt_path, index=False)
print(f"Video {n_total} khung -> {len(idx)} ảnh mẫu (dev={n_dev}, test={len(idx)-n_dev}), nền từ {len(bg_frames)} khung.")
