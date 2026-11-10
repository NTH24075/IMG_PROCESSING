# Tuần 1 — kế hoạch triển khai đã được người dùng yêu cầu

Phạm vi kế thừa hai bản kế hoạch v2. Không chạy đánh giá test; không giả lập xác nhận giảng viên hoặc người kiểm nhãn thứ hai.

1. Lưu mã/nhãn/nền và tái lập baseline 26 dev trước khi tách module.
2. Pipeline dùng chung, cấu hình TOML, preprocessing/segmentation/regions, kiểm tra bằng unittest.
3. Chuẩn bị dữ liệu nhiều cảnh an toàn: giữ vtest/holdout cũ; Meet_Crowd dev; WalkByShop test; nền từ prefix cố định; nhãn XML là draft cho đến rà soát.
4. Predict ảnh không nhãn + GUI Tk cục bộ; annotate nhiều cảnh để rà soát có provenance.
5. Evaluator dev, kiểm tra test/freeze/audit gồm all/manifest/wrapper; không tạo freeze thật tuần 1.
6. Kiểm tra tích hợp, enhancement/filter benchmark, thử watershed trên dev, báo cáo checkpoint và phần còn cần người dùng/nhóm.

Snapshot trước thay đổi: work/week1_before của chat. Chưa có Git repository; không tự tạo commit giả danh người dùng.
