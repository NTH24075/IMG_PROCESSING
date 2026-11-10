# Nguồn dữ liệu dự án

| Nguồn | Vai trò | Đường dẫn |
|---|---|---|
| OpenCV sample vtest.avi đang có | scene_01 dev và holdout lịch sử | data/raw/vtest.avi |
| CAVIAR Meet_Crowd | scene_02 dev | data/raw/caviar/Meet_Crowd.mpg |
| CAVIAR WalkByShop1cor | scene_03 test | data/raw/caviar/WalkByShop1cor.mpg |

Nguồn chính thức CAVIAR và điều kiện dùng: https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/CAVIARDATA1/ . Ghi nhận EC Funded CAVIAR project/IST 2001 37540, theo CC BY-SA ở trang nguồn. URL video/XML, SHA-256 và số khung thực nằm trong data/raw/caviar/SOURCE.md và metadata.json. Nguồn vtest ban đầu cần nhóm kiểm tra điều kiện và trích dẫn đúng khi nộp, không tự đoán giấy phép.

Mỗi nguồn chỉ là tài sản đầu vào, không đồng nghĩa nhãn đã được rà soát. Bộ mới ở data/week1/; nhãn XML được nhập ban đầu dưới dạng draft và giữ provenance; đến ngày kiểm chứng 30/10/2026, toàn bộ 151 nhãn chính đã được người kiểm tra chéo (peer_reviewed). Nền ứng viên từ prefix; Chủ dự án đã chốt ROI scene_03 x=0..383, y=65..287 ngày 28/10/2026; nền hiện đã reviewed và nhãn đã peer_reviewed theo ROI này. Ảnh hiệu chuẩn độc lập thử trong raw/caviar/calibration bị loại do có người, không dùng trong pipeline và không là cảnh bổ sung.
