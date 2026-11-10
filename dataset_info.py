"""Sinh thống kê dữ liệu cho Bước 3 của báo cáo -> data/dataset_info.md"""
import cv2, pandas as pd, numpy as np
df = pd.read_csv("data/ground_truth.csv"); h, w = cv2.imread("data/background.png").shape[:2]
L = ["# Mô tả dữ liệu", "",
     "- Nguồn: video camera cố định `vtest.avi` đi kèm OpenCV (samples/data). **Tự kiểm tra nguồn gốc gốc của video và giấy phép trước khi ghi vào báo cáo.**",
     f"- Video gốc: 795 khung, 10 fps, {w}x{h}, camera cố định, cảnh ngoài trời có người đi bộ.",
     f"- Lấy mẫu: {len(df)} ảnh (mỗi 15 khung ≈ 1,5 s). Chia theo thời gian: dev = nửa đầu video, test = nửa sau (không trộn ngẫu nhiên để tránh khung gần nhau rơi vào cả hai tập).",
     "- Ảnh nền: trung vị theo thời gian của các khung (không dùng nhãn).",
     "- Nhãn: số người trong ảnh, gán tay bằng `annotate.py` (click từng người); tọa độ lưu trong `annotations.json`.", "",
     "| Tập | Số ảnh | Đã gán nhãn | Đếm TB | Min | Max |", "|---|---|---|---|---|---|"]
for s in ["dev", "test"]:
    d = df[df.split == s]; c = d["count"].dropna()
    L.append(f"| {s} | {len(d)} | {len(c)} | " + (f"{c.mean():.2f} | {int(c.min())} | {int(c.max())} |" if len(c) else "- | - | - |"))
open("data/dataset_info.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
