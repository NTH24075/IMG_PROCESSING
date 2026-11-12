# Object Counting in Surveillance Images — v3

Hệ thống xử lý ảnh cổ điển, dùng camera cố định và nền tương ứng. Video màu được lấy mẫu; pipeline chính chuyển grayscale → enhancement/filter → trừ nền → threshold → morphology → contour → số đếm. Giữ ảnh màu để hiển thị. Không huấn luyện deep learning.

## Bắt đầu

```bash
cd /home/nth24075/Documents/IMG_processing/object_counting_gd1_2/oc2
uv sync --locked
uv run python app.py
```

Demo Tk cục bộ: chọn scene_01 hoặc scene_02, chọn ảnh đúng camera rồi bấm **Đếm ảnh**. Ctrl+O chọn ảnh; Ctrl+Enter chạy. Không cần nhãn. Ảnh test (scene_03, holdout vtest) chỉ chạy sau tag freeze-test.

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

run_baseline.py là wrapper tương thích, cùng freeze guard. Không dùng --frames/--gt/--bg cũ để tạo một evaluator khác; dùng manifest nhiều cảnh. Mỗi run có predictions.csv, metrics.json, environment.json, config.toml và ví dụ ảnh. Có ảnh lỗi thì CLI trả mã lỗi, không coi là đánh giá hoàn chỉnh.

## v3: C1–C3, benchmark độ bền, chẩn đoán lỗi

Pipeline v3 thêm cờ trong `[processing]` (mặc định = baseline, test hồi quy 26 dev giữ nguyên):

| Cờ | Giá trị | Đóng góp |
|---|---|---|
| `enhancement` | none, gamma, percentile, equalize_separate, **match1**, **match2** | C1 histogram matching I→B (2 lượt: bỏ foreground thô khỏi CDF) |
| `filter` | none, mean, gaussian, median (cả backend scratch) | E3 |
| `shadow` (+ `shadow_lo/hi/t_grad/k`) | true/false | C2 tỉ số I/B + |ΔSobel| |
| `fill_holes`, `labeling` (bfs, twopass) | | Lấp lỗ; gán nhãn 3 lượt + union-find (backend scratch) |
| `count_method` (+ `cap`, `head_vote`) | blob, **area**, area_hough; watershed = `split_merged=true` | C3 mô hình diện tích theo phối cảnh + Hough đầu |

Mô hình phối cảnh nằm ở `[scene.perspective]` trong `configs/scenes/scene_0x.toml`, chỉ fit từ dev:

```bash
uv run python calibrate_scene.py            # scene_01, scene_02; scene_03 không có dev nên C3 tự lùi về 1 blob = 1 người
uv run python make_robustness.py            # data/week1/manifests/robustness_{dev,test}.csv; ảnh suy giảm sinh trong bộ nhớ
uv run python run_eval.py --experiment configs/experiments/e5_merged.toml   # mỗi biến thể một run + summary.csv + figures/
uv run python run_eval.py --manifest data/week1/manifests/robustness_dev.csv --config configs/final.toml
uv run python run_benchmark.py              # E6 tự cài vs OpenCV
```

Thí nghiệm: `e2_lighting`, `e3_noise`, `e4_shadow`, `e5_merged`, `e7_ablation`, `e8_robustness_final` trong configs/experiments/. Mỗi run ghi config.toml đã giải quyết, git.json, predictions.csv, metrics.json (by_scene, by_condition), error_types.csv (merge/split/false_blob/miss từ điểm click vtest và tâm box CAVIAR), environment.json. Kết quả dev ngày 12/11/2026 và lý do chọn final: docs/decisions.md.

## Test: freeze nhẹ bằng git tag

Mọi manifest chứa image_id/sha256 của test.csv hoặc legacy_holdout.csv (kể cả robustness_test, manifest ghép, ảnh đổi tên, predict.py) chỉ chạy khi: working tree sạch, HEAD trùng tag `freeze-test` (hoặc `freeze-test-N` mới nhất), config đã commit. Một dòng được append vào outputs/test_runs.log **trước** khi đọc ảnh test; không ghi được log thì không chạy. Đây là rào chắn chống chạy nhầm, không phải bảo mật.

```bash
# Cuối ngày 10, sau khi nhóm chốt final.toml (hiện là PROPOSAL):
git add -A && git commit -m "Freeze baseline + final"
git tag freeze-test
uv run python run_eval.py --config configs/baseline.toml --split test
uv run python run_eval.py --config configs/final.toml --split test
uv run python run_eval.py --config configs/final.toml --split legacy_holdout
uv run python run_eval.py --config configs/final.toml --manifest data/week1/manifests/robustness_test.csv
```

Sửa lỗi sau test: commit, `git tag freeze-test-2`, ghi vào docs/decisions.md, báo cáo nêu số lần chạy test.

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

Kiểm tra core, preprocessing/matching, degrade, shadow, gán nhãn 3 lượt, phối cảnh/Hough, loại lỗi, freeze guard (repo git giả trong thư mục tạm) và hồi quy toàn bộ 26 dev của cả hai backend. Test hồi quy cần data vtest sẵn có.

- object_counting/: pipeline, IO, degrade, preprocessing, segmentation, regions, counting, metrics, experiments (runner + freeze), visualization, annotation.
- configs/: baseline, final (proposal), experiments/eN.toml, scenes/ (kèm phối cảnh).
- data/week1/: manifests, frames, nền ứng viên, nhãn có provenance.
- tests/: unittest và bảng baseline chụp trước khi chuyển mã.
- docs/: annotation_guide.md, decisions.md.
- tasks/: plan.md, todo.md — trạng thái thật, có mục cần người nhóm.
- reports/: nháp phương pháp và bảng/hình tuần 1.

## Việc cần nhóm hoàn tất

Checkpoint nhãn/nền và phần mềm đã đạt sau kiểm chứng ngày 30/10/2026: 151 nhãn chính peer_reviewed, hai nền mới reviewed, dữ liệu pass và 16 unittest qua. Còn xác nhận với giảng viên phạm vi người+cần nền, 5 bước, bài báo và hạn nộp. Tiếp theo chẩn đoán lỗi dev, thử cải tiến bù sáng/segmentation theo bằng chứng và đánh giá lại các biến thể; chưa chọn final hoặc chạy test.

Tài liệu chính: hai bản kế hoạch v2 trong thư mục IMG_processing và báo cáo trạng thái tuần 1 xuất kèm chat.
