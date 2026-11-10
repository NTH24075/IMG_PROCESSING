# Object Counting in Surveillance Images — tuần 1

Hệ thống xử lý ảnh cổ điển, dùng camera cố định và nền tương ứng. Video màu được lấy mẫu; pipeline chính chuyển grayscale → enhancement/filter → trừ nền → threshold → morphology → contour → số đếm. Giữ ảnh màu để hiển thị. Không huấn luyện deep learning.

## Bắt đầu

```bash
cd /home/nth24075/Documents/IMG_processing/object_counting_gd1_2/oc2
uv sync --locked
uv run python app.py
```

Demo Tk cục bộ: chọn scene_01 hoặc scene_02, chọn ảnh đúng camera rồi bấm **Đếm ảnh**. Ctrl+O chọn ảnh; Ctrl+Enter chạy. Không cần nhãn. scene_03 được giữ test và bị khóa trước freeze.

Ví dụ ảnh mới không nhãn:

```bash
uv run python predict.py --scene scene_02 --image data/week1/frames/scene_02/f00180.png
```

Ảnh/nền phải cùng kích thước và góc camera. Kết quả ở outputs/predictions/<run_id>/ gồm result.json, overlay.png, panel.png và stages/. Không dùng số đếm dự đoán làm nhãn thật.

## Dữ liệu đã chuẩn bị

| Cảnh | Nguồn | Số ảnh bộ chính | Vai trò |
|---|---|---:|---|
| scene_01 | vtest.avi | 26 | Dev, đã kiểm tra chéo nhãn |
| scene_02 | Meet_Crowd.mpg | 50 | Dev, nhãn đã được người rà soát và kiểm tra chéo |
| scene_03 | WalkByShop1cor.mpg | 75 | Test chuyển cảnh, nhãn đã được người rà soát và kiểm tra chéo |

Tổng **151 ảnh chính**, cộng 27 holdout vtest lịch sử giữ riêng. data/week1/ chứa dữ liệu mới để không ghi đè data/frames/, ground_truth.csv, annotations.json và background.png cũ. prepare_week1.py chỉ tạo một lần, từ chối ghi đè bộ đang có.

```bash
uv run python validate_dataset.py
```

Nguồn/nền: data/SOURCES.md và data/raw/caviar/SOURCE.md. Scene config ở configs/scenes/. scene_02 dùng prefix 30 khung, scene_03 prefix 500 khung; lấy nền mỗi 5 khung, không dùng khung đánh giá tương lai. Nền scene_02 và scene_03 đã được người kiểm tra xác nhận reviewed; kiểm chứng trạng thái ngày 30/10/2026. Chủ dự án đã chốt ROI scene_03 vùng hành lang gần (x=0..383, y=65..287) ngày 28/10/2026 do nền vùng xa còn người; nhãn và nền đã được rà soát theo ROI này. Không tuyên bố đếm toàn khung ở cảnh này. Các clip hiệu chuẩn độc lập đã thử có người, không được dùng làm nền sạch.

Vtest vẫn dùng nền trung vị ngoại tuyến lịch sử từ toàn video. Thiết lập này phải được khai báo, không dùng để tuyên bố nền causal/online. source_frame và timestamp_seconds của MPEG là chỉ số đọc tuần tự và thời gian danh nghĩa theo FPS, không bảo đảm tự căn chỉnh XML/PTS.

## Rà soát nhãn và nền

Đọc docs/annotation_guide.md. Nhãn XML **không tự trở thành ground truth đã kiểm chứng**. Thay Ten_that bằng người đang kiểm tra; hai lượt dùng hai người khác nhau:

```bash
uv run python review_labels.py --scene scene_02 --reviewer "Ten_that"
uv run python review_labels.py --scene scene_03 --reviewer "Ten_that"
uv run python review_labels.py --scene scene_01 --reviewer "Ten_that"
uv run python review_background.py --scene scene_02 --reviewer "Ten_that"
uv run python review_background.py --scene scene_03 --reviewer "Ten_that"
```

Công cụ nhãn không chạy đếm. Lượt đầu tạo reviewed; người khác xác nhận cùng số tạo peer_reviewed. Sửa số phải kiểm tra chéo lại. Có backups và review_log.jsonl; không dùng cùng một người với hai tên để tạo duyệt giả. Holdout lịch sử không hiện mặc định; --include-legacy chỉ khi thật sự cần.

## Đánh giá dev

```bash
uv run python run_eval.py --config configs/baseline.toml --split dev
uv run python run_eval.py --config configs/baseline.toml --split dev --compare-scratch
```

Baseline đã xử lý thành công 76/76 ảnh dev, không lỗi; toàn bộ 151 nhãn bộ chính đã peer_reviewed. Run outputs/experiments/20261029T171005Z_8392ad33ef cho MAE=1.000, RMSE=1.348, exact-count Acc=38.16%, Bias=-0.947 trên 76 dev. Đối chiếu số nhãn và tính lại chỉ số ngày 30/10/2026 khớp kết quả đã lưu. Đây là mốc baseline, chưa phải kết quả test hoặc cấu hình cuối. --include-draft chỉ dùng cho dev, phải gọi kết quả là tạm thời nếu có draft trong bộ dữ liệu khác.

run_baseline.py là wrapper tương thích, có cùng gate. Không dùng --frames/--gt/--bg cũ để tạo một evaluator khác; dùng manifest nhiều cảnh. Mỗi run có predictions.csv, metrics.json, environment.json, config.toml và ví dụ ảnh. Có ảnh lỗi thì CLI trả mã lỗi, không coi là đánh giá hoàn chỉnh.

## Test đang khóa

Không có configs/test_freeze.json thật trong tuần 1. --split test, --split all, manifest chứa test, ảnh test đổi tên nguyên nội dung và chọn camera test trong demo bị kiểm tra. Nhật ký outputs/test_audit.jsonl ghi request/allowed/denied/completed/failed.

Tuần 2, sau khi nhãn chính peer_reviewed và nền được duyệt theo ROI đã chốt, TV1 + TV5 mới đóng băng cả baseline và final:

```bash
# Chỉ chạy ở mốc cuối ngày 10, sau rà soát và chọn final thật:
uv run python freeze_test.py --configs configs/baseline.toml configs/final.toml --reviewers "Nguoi_1" "Nguoi_2"
uv run python run_eval.py --config configs/baseline.toml --split test
uv run python run_eval.py --config configs/final.toml --split test
```

final.toml chưa được tạo vì tuần 1 chưa chọn final. Gate gắn hash mã/cấu hình/manifest/ảnh/nền/nhãn, không chỉ tên final hay cờ boolean. Đổi snapshot bị từ chối. Đây là rào chắn chống chạy nhầm, không chống người cố ý sửa mã hoặc log. Holdout lịch sử nằm ngoài freeze chính và cần protocol riêng. Không tuning theo test.

## Khảo sát tuần 1

```bash
uv run python week1_experiments.py
```

Các nhánh: baseline, gamma, percentile, median, fixed threshold, bỏ morphology, watershed. Tất cả chạy dev. Bảng khảo sát 7 biến thể lưu từ ngày 28/10 chỉ dùng 26 nhãn cũ; chưa chạy lại đủ các biến thể trên 76 nhãn dev hiện đã kiểm tra chéo. Median có kết quả sơ bộ tốt hơn baseline trên 26 ảnh, nhưng chưa chọn final. Watershed chưa cải thiện trên phần này nên giữ nhánh thử nghiệm, mặc định tắt.

Có CSV thời gian OpenCV/NumPy trên 4 ảnh dev cố định, seed=42 cho nhiễu, hình Gaussian/median và minh họa Sobel/Canny. Sobel/Canny không phải một nhánh đếm riêng. Không gọi độ đồng thuận mask giữa hai bản cài đặt là độ chính xác phân đoạn ground truth.

## Kiểm tra và cấu trúc

```bash
uv run python -m unittest discover -s tests -v
```

Kiểm tra core, nhãn/review, ROI, hai đối tượng tổng hợp, gate/freeze/audit và hồi quy toàn bộ 26 dev của cả hai backend. Test hồi quy cần data vtest sẵn có. Các test freeze dùng dữ liệu giả trong thư mục tạm; không phải chạy test thật.

- object_counting/: pipeline, IO, preprocessing, segmentation, regions, metrics, evaluation guard, experiments, visualization, annotation.
- configs/: phương pháp và cảnh; trạng thái phát triển, chưa có final/freeze.
- data/week1/: manifests, frames, nền ứng viên, nhãn có provenance.
- tests/: unittest và bảng baseline chụp trước khi chuyển mã.
- docs/: annotation_guide.md, decisions.md.
- tasks/: plan.md, todo.md — trạng thái thật, có mục cần người nhóm.
- reports/: nháp phương pháp và bảng/hình tuần 1.

Thư mục chưa có Git repository; đã giữ snapshot trước thay đổi trong work/week1_before của chat. Dùng quy trình Git của nhóm khi đưa mã vào repo chung, không có commit giả danh thành viên.

## Việc cần nhóm hoàn tất

Checkpoint nhãn/nền và phần mềm đã đạt sau kiểm chứng ngày 30/10/2026: 151 nhãn chính peer_reviewed, hai nền mới reviewed, dữ liệu pass và 16 unittest qua. Còn xác nhận với giảng viên phạm vi người+cần nền, 5 bước, bài báo và hạn nộp. Tiếp theo chẩn đoán lỗi dev, thử cải tiến bù sáng/segmentation theo bằng chứng và đánh giá lại các biến thể; chưa chọn final hoặc chạy test.

Tài liệu chính: hai bản kế hoạch v2 trong thư mục IMG_processing và báo cáo trạng thái tuần 1 xuất kèm chat.
