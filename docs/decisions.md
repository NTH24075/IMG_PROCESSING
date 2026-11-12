# Quyết định và trạng thái tuần 1

- Người dùng yêu cầu triển khai tuần 1 theo kế hoạch v2, giữ test chuyển cảnh.
- Dùng 3 nguồn đã chọn: vtest + Meet_Crowd dev; WalkByShop1cor test. Không chạy đánh giá test trong tuần 1.
- Dữ liệu mới nằm trong data/week1/ để không ghi đè frames/ground_truth/background lịch sử. Đổi này tránh di chuyển dữ liệu đang dùng trước khi xác minh mapping.
- Baseline đã tái lập trên 26 dev trước khi tách; pipeline mới cho số đếm giống hệt ở cả OpenCV và NumPy. Snapshot mã/nhãn/nền cũ được giữ trong work/week1_before của chat.
- Chỉ dùng prefix đã định để tạo nền ứng viên: scene_02 [0,30), scene_03 [0,500), lấy mỗi 5 frame. Sampling bắt đầu sau prefix. Nền hành lang có dấu vết người ở xa, chưa được nhóm duyệt; không lấy thêm khung test để làm sạch.
- Nhãn 125 ảnh CAVIAR là draft_xml; chưa có kiểm tra chéo. Không giả danh thành viên nhóm để tự đặt peer_reviewed.
- Dùng Tk/ttk cho demo desktop và rà soát vì Python hiện tại đã có Tk; không thêm thư viện web hoặc dependency mới. Một cửa sổ, điều khiển có nhãn, chọn file, thông báo lỗi/rỗng/thành công, Ctrl+O/Ctrl+Enter; GUI không giữ thuật toán riêng.
- Pipeline chủ đạo grayscale; giữ ảnh màu cho overlay. LUT enhancement dùng chung cho ảnh/nền; percentile LUT ước lượng từ nền.
- Scratch backend triển khai grayscale/Gaussian/Otsu/morphology/BFS; median/mean là nhánh OpenCV, không gọi đó là tự cài NumPy.
- Watershed là thử nghiệm OpenCV, mặc định tắt. Khảo sát sơ bộ trên 26 nhãn dev cũ chưa cải thiện MAE; không chốt làm mặc định khi cảnh mới chưa có nhãn kiểm chứng.
- Test gate khóa CLI, wrapper, all, manifest chứa test, ảnh test đổi tên nguyên nội dung và demo camera test. Đây là rào chắn vận hành, không chống người cố ý sửa mã/dữ liệu.
- Không tạo test_freeze.json thật trong tuần 1. Unit test freeze dùng dữ liệu giả ở thư mục tạm; không phải đánh giá test thật.
- Cần nhóm/giảng viên xác nhận: phạm vi người+cần nền, nguyên văn 5 bước và bài báo, ngày/giờ nộp, phụ lục. Chưa có xác nhận; không ghi là hoàn tất.

- Sau kiểm tra nền/warmup, đề xuất ROI scene_03 y=65..287 trước khi đánh giá test. Nền trung vị còn người ở xa; ảnh hiệu chuẩn độc lập thử cũng có người nên bị loại. Không dùng khung đánh giá để thay nền. ROI mới là đề xuất cần nhóm duyệt; XML draft theo tâm box trong ROI. Không công bố đếm toàn ảnh cho scene_03.

- 28/10/2026 — Chủ dự án chấp thuận: “Dùng ROI hành lang gần”. Quyết định này thay thế trạng thái đề xuất ở mục trên: scene_03 đếm trong x=0..383, y=65..287 của ảnh 384×288. Cấu hình ghi người duyệt theo vai trò project_owner, không suy diễn rằng cả nhóm đã rà soát. Giới hạn vùng xa nhằm tránh người còn trong nền khởi tạo; chưa dùng sai số test để chọn ROI. Giữ background_status=needs_visual_review và nhãn mới draft_xml cho đến khi có người kiểm tra thật. Không đổi tọa độ ROI hoặc nhãn nháp đã chuẩn bị.

- Background review 2026-10-29T13:25:40.377281+00:00: scene_03, reviewer=NTH, status=reviewed.

- Background review 2026-10-29T13:26:18.350776+00:00: scene_03, reviewer=NTH, status=reviewed.

- Background review 2026-10-29T13:26:32.190670+00:00: scene_02, reviewer=NTH, status=reviewed.

- Kiểm chứng checkpoint 30/10/2026: dữ liệu pass; 151 ảnh chính peer_reviewed với hai tên người duyệt khác nhau được ghi nhận, 27 holdout lịch sử giữ riêng. Nền scene_02/03 reviewed. Tính lại số liệu từ run 20261029T171005Z_8392ad33ef và đối chiếu counts.csv: đủ 76 dev, MAE=1.000, RMSE=1.348, exact-count Acc=38.16%, Bias=-0.947; config snapshot khớp baseline hiện tại. Chạy lại 16 unittest thành công. Checkpoint dữ liệu/phần mềm đạt; xác nhận yêu cầu giảng viên còn mở. Không tạo final/freeze hoặc chạy đánh giá test; bù sáng/HOG mới được thảo luận, chưa triển khai. Các mục draft/chờ duyệt ở trên là lịch sử trước khi người dùng hoàn tất rà soát.

# v3 — triển khai mã (12/11/2026, nhánh v3-roadmap)

- Thay gate hash/snapshot/test_audit.jsonl của v2 bằng freeze nhẹ (v3 §6): `experiments.check_freeze` yêu cầu cây sạch, HEAD = tag `freeze-test[-N]` mới nhất, config đã commit; nhận diện test theo image_id/source_image_id/sha256 của test.csv + legacy_holdout.csv; append outputs/test_runs.log trước khi đọc ảnh test. Đã xóa evaluation_guard.py, freeze_test.py, test_evaluation_guard.py. Chưa tạo tag, chưa chạy test.
- Giữ bố cục dữ liệu data/week1/ (không di chuyển sang data/manifests/ như bản cấu trúc v3) để không phá đường dẫn/sha256 đang dùng. Config giữ một bảng `[processing]` phẳng thay vì các bảng [normalize]/[filter]…; mọi cờ v3 mặc định bằng baseline.
- Điểm cho chẩn đoán lỗi/hiệu chỉnh: click vtest (data/annotations.json) + tâm box XML CAVIAR (xml_candidates.json), lọc theo ROI. Điểm thuộc blob khi nằm trong bbox nới 3 px (shortcut, ghi trong metrics.py).
- Phối cảnh fit chỉ từ dev (calibrate_scene.py): scene_01 k=0.1187, c=−0.126 (125 blob 1 người); scene_02 k=0.0278, c=17.51 (38 blob). head_alpha chọn theo MAE dev (các α 0.10–0.15 hòa; lấy 0.10). scene_03 không có ảnh dev nên **không có mô hình phối cảnh**: C3 tự lùi về 1 blob = 1 người trên test chuyển cảnh. Báo cáo phải nêu giới hạn này.
- E1: baseline 76 dev MAE=1.000, Bias=−0.947, khớp checkpoint 30/10; hồi quy 26 dev cả hai backend vẫn qua.
- E5 (run 20261112T134922Z_fc6328119a): blob 1.000 · area 0.645 · area_hough 0.645 · watershed 1.026. Lỗi merge 65→30 với mô hình diện tích. Hough đầu không thay đổi kết quả nào trên dev (đầu chỉ ~3–5 px ở vtest); giữ như thí nghiệm, không bật trong final.
- E4: shadow trên baseline 1.000→0.987; trên nền final (match2 + area) 0.605→0.592, merge 30→26.
- E2 (run 20261112T134725Z_11a847a6fc): không chuẩn hóa MAE lên 13.5 (contrast 0.3) / 4.2 (gain 0.6); match2 giữ 0.99–1.03 ở mọi mức, match1 1.01–1.05. equalize_separate **không tệ như dự đoán** (1.08 ổn định) — ghi đúng như vậy trong báo cáo.
- E3 (run 20261112T134816Z_85dd136fde): nhiễu Gauss gần như không ảnh hưởng baseline; muối tiêu 0.05 tăng MAE lên 1.21; median5 giữ ≤1.01.
- E7 (run 20261112T135047Z_1ed6eb9307), final đề xuất = match2 + shadow + area: MAE 0.592, Bias −0.487; bỏ match 0.632, bỏ shadow 0.605, bỏ area 1.000; thêm Hough 0.592 (không đổi); thêm median 0.632.
- Độ bền final (run 20261112T135149Z_cea5afcc8d, robustness_dev): MAE tệ nhất theo loại — final: contrast 0.61, gain 0.59, gamma 0.66, gauss 0.63, **saltpepper 1.63**; baseline: 13.47 / 4.20 / 2.22 / 1.05 / 1.14. final+median: saltpepper 0.78 nhưng dev sạch 0.632. **Cần nhóm chốt ở mốc ngày 7**: ưu tiên dev sạch (final hiện tại) hay độ bền muối tiêu (thêm `filter = "median"`).
- E6 (outputs/benchmarks/20261112T135315Z_ad7fbfccb2): bản tự cài khớp OpenCV ≥99% pixel (Gaussian 0.990 do kernel), chậm hơn 2–200×; Hough tự cài ~386 ms/ảnh so với cv2.HoughCircles ~3 ms (khác thuật toán, chỉ so số lượng/thời gian).
- Tất cả số trên là dev; perspective, α và final đều chọn trên dev nên MAE dev lạc quan. Số báo cáo cuối phải lấy từ test sau freeze.
