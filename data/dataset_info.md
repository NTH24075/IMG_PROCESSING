# Mô tả dữ liệu

- Nguồn: video camera cố định `vtest.avi` đi kèm OpenCV (samples/data). **Tự kiểm tra nguồn gốc gốc của video và giấy phép trước khi ghi vào báo cáo.**
- Video gốc: 795 khung, 10 fps, 768x576, camera cố định, cảnh ngoài trời có người đi bộ.
- Lấy mẫu: 53 ảnh (mỗi 15 khung ≈ 1,5 s). Chia theo thời gian: dev = nửa đầu video, test = nửa sau (không trộn ngẫu nhiên để tránh khung gần nhau rơi vào cả hai tập).
- Ảnh nền: trung vị theo thời gian của các khung (không dùng nhãn).
- Nhãn: số người trong ảnh, gán tay bằng `annotate.py` (click từng người); tọa độ lưu trong `annotations.json`.

| Tập | Số ảnh | Đã gán nhãn | Đếm TB | Min | Max |
|---|---|---|---|---|---|
| dev | 26 | 26 | 6.38 | 3 | 9 |
| test | 27 | 27 | 5.93 | 2 | 9 |
