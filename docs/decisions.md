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

# Thống nhất quy tắc đếm ở biên ROI (15/11/2026, trước freeze, chưa xem test)

- Trước đây ba bên dùng ba quy tắc: nhãn = tâm phần cơ thể nhìn thấy trong ROI (annotation_guide); contour = cắt mặt nạ theo ROI (người chỉ một phần trong ROI vẫn được đếm, diện tích bị cắt làm lệch mô hình phối cảnh); HOG = điểm chân trong ROI.
- Nay contour và HOG cùng theo quy tắc của nhãn: đếm vùng/khung có **tâm bbox** nằm trong ROI. Contour không cắt mặt nạ nữa nên vùng vắt qua biên giữ nguyên diện tích. Test: tests/test_core.py::test_roi_edge_uses_body_centre.
- Chỉ ảnh hưởng scene_03 (scene_01/02 dùng toàn ảnh): dev không đổi — final MAE 0.592 (run 20261115T065548Z_624027628a), HOG 1.500. Ảnh hưởng trên scene_03 chưa đo vì scene_03 chỉ có ảnh test.
- Rủi ro đã biết: ngoài ROI scene_03 nền còn người, tiền cảnh rác ở vùng xa có thể dính vào người ở sát biên ROI và kéo tâm ra ngoài. Quyết định dựa trên định nghĩa nhãn, không dựa trên kết quả test.

# Khoảng tin cậy bootstrap khối (15/11/2026)

- metrics_with_ci / paired_mae_difference (metrics.py): khối 5 ảnh liên tiếp trong cùng cảnh, 2000 lần, seed 0. Bootstrap từng ảnh trên dev cho MAE contour [0.42, 0.78]; bootstrap khối rộng hơn [0.26, 0.97] vì tính đến tương quan giữa các khung kề nhau — dùng bootstrap khối trong báo cáo.
- HOG − contour (dev, ghép cặp): 0.91 [0.33, 1.45] → contour tốt hơn có ý nghĩa trên dev.
- E7 (run 20261115T070203Z_c0a1b3b924): chỉ mô hình diện tích (no_area 1.000 [0.63, 1.37] so với final 0.592 [0.26, 0.97]) tách rõ; khác biệt của match2, shadow, median (0.59–0.63) nằm gọn trong sai số ngẫu nhiên của 76 ảnh dev, không được viết là cải thiện có ý nghĩa.

# Đồng bộ code với báo cáo và freeze (21/11/2026)

- Đối chiếu báo cáo PDF (bao_cao_object_counting.pdf) với code; sửa các chỗ báo cáo có số nhưng code chưa sinh ra:
  - `metrics.error_types` thêm loại **gán thừa** (`over_assign` = max(0, n−m) trên vùng có m≥1). Năm loại khớp Bảng 5/7; đẳng thức sai số có dấu có unit test.
  - `run_c3_cv.py`: kiểm định chéo K khối thời gian liên tiếp cho C3, chỉ trên dev. K=5: MAE 0.868 [0.49, 1.26]; so với trong mẫu +0.276 [0.07, 0.53]; so với cơ sở +0.132 [−0.03, 0.29]; vtest p 0.117–0.122 (93–107 vùng), Meet_Crowd p 0.011–0.053 (21–38 vùng). Run 20261121T033819Z_9144319d2f.
  - `run_bootstrap_sweep.py`: KTC MAE với khối 1/3/5/10 = [0.43, 0.78] / [0.33, 0.88] / [0.26, 0.97] / [0.24, 1.07]; giữ khối 5.
  - `compare_runs.py`: ∆MAE ghép cặp giữa hai lần chạy đã có (không xử lý lại ảnh).
- `predict.py` và `app.py` mặc định dùng `configs/final.toml` (trước là baseline, nên demo không chạy C1–C3 như báo cáo mô tả).
- `make_robustness.py`: `robustness_test.csv` chỉ gồm 75 ảnh scene_03 × 18 điều kiện; bỏ 27 ảnh holdout vtest, vì báo cáo không dùng chúng và C3 vẫn bật trên vtest. Quyết định dựa trên phạm vi báo cáo, chưa xem kết quả test.
- E4 thêm biến thể `on_hi085` (chỉ đổi cận trên 0.85): MAE 0.987, giống cấu hình mặc định. MAE 1.000 chỉ xuất hiện khi đổi thêm t_grad=15 (`on_strict`). **Báo cáo §3.4 phải sửa câu "Với cận trên 0,85, MAE trở lại 1,000".**
- E6 thêm cột trùng tuyệt đối: mọi thao tác trùng 100% với OpenCV, trừ Gauss 5×5 (85.5% trùng tuyệt đối, ≥98.5% trong ±1). Tham chiếu của so khớp histogram là NumPy np.interp, vì OpenCV không có hàm này.
- Kiểm tra lại toàn bộ trên dev sau khi sửa: Bảng 5, 6 (dòng phát triển), 7 không đổi so với báo cáo.
- `final.toml` chuyển sang `status = "frozen"`; cấu hình không đổi (match2 + shadow + area). Tag `freeze-test` đặt sau commit này, rồi mới chạy test một lần.
