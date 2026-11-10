# Bàn giao tuần 1 — Object Counting

Cập nhật và kiểm chứng: 30/10/2026. Trạng thái: **checkpoint dữ liệu và phần mềm tuần 1 đã đạt; phần xác nhận yêu cầu môn học còn mở. Baseline chưa đạt mức đếm đáng tin cậy và chưa đánh giá test.**

## 1. Đã thực hiện

- Snapshot mã, nhãn và nền trước thay đổi; tái lập baseline cũ trước tách module.
- Một pipeline dùng chung cho predict, GUI và batch evaluator; không nhận ground truth khi suy luận.
- Cấu hình TOML; enhancement gamma/percentile, Gaussian/median/mean, trừ nền, Otsu/ngưỡng cố định, morphology, contour; backend NumPy và nhánh watershed có giới hạn.
- Demo Tk cục bộ, công cụ kiểm nhãn theo người duyệt, công cụ duyệt nền; không thêm dependency.
- Dataset chính 151 ảnh: 76 dev, 75 test. 27 holdout vtest cũ giữ ngoài đánh giá chính.
- Nhập XML thành nhãn nháp có provenance; không giả lập người duyệt hoặc kiểm tra chéo.
- Evaluator chỉ dùng nhãn người/đã duyệt theo mặc định, báo coverage; không tự tính XML draft vào chỉ số chính.
- Gate/freeze/audit cho test, all, wrapper cũ, manifest test khai là dev và ảnh test đổi tên nguyên nội dung. Demo cảnh test cũng bị khóa.
- Khảo sát 7 biến thể trên dev, benchmark 4 ảnh dev cố định, minh họa nhiễu seed=42 và Sobel/Canny.

## 2. Kiểm chứng

- 16 unittest đã chạy thành công; có ví dụ tính tay, ảnh tổng hợp, lỗi đầu vào và guard/freeze dữ liệu giả.
- Hồi quy: số đếm cả 26 dev khớp snapshot cũ cho cả OpenCV và NumPy.
- Đã đảo điều kiện guard để kiểm tra độ nhạy test: bộ kiểm tra phát hiện lỗi; sau đó khôi phục mã đúng.
- CLI dự đoán ảnh không nhãn và callback GUI chạy thật; thử lỗi đường dẫn, trạng thái rỗng/lỗi/thành công và mở giao diện kiểm nhãn mà không lưu duyệt giả.
- Đánh giá baseline xử lý 76/76 ảnh dev, không lỗi; hiện dùng đủ 76 nhãn dev đã peer_reviewed. Đã tính lại chỉ số từ predictions.csv và đối chiếu với counts.csv hiện tại; khớp kết quả đã lưu.
- Lệnh test, all, manifest test giả dev bị từ chối trước xử lý ảnh; có audit. Chưa tạo freeze thật, chưa có đánh giá test thật.
- Compile Python và kiểm tra tương thích dependency thành công.
- Hash nhãn ground_truth.csv, annotations.json và background.png vtest cũ được giữ nguyên.

## 3. Tình trạng dữ liệu

| Thành phần | Trạng thái |
|---|---|
| vtest dev | 26 ảnh, peer_reviewed |
| Meet_Crowd dev | 50 ảnh, đã rà soát bằng người và peer_reviewed |
| WalkByShop test | 75 ảnh, đã rà soát bằng người và peer_reviewed; chưa chạy đếm test |
| Holdout lịch sử | 27 ảnh, giữ riêng và chưa dùng đánh giá chính |
| Nền scene_02 | Tạo từ prefix [0,30), trạng thái reviewed |
| Nền scene_03 | Tạo từ prefix [0,500), reviewed trong ROI hành lang gần; vùng xa ngoài phạm vi đếm |
| ROI scene_03 | Chủ dự án đã chốt x=0..383, y=65..287 (hành lang gần), ngày 28/10/2026; nền đã reviewed và nhãn đã peer_reviewed theo ROI này |

ROI được chủ dự án chốt sau kiểm tra nền/warmup, trước khi xem kết quả test. Scene_03 chỉ đếm vùng hành lang gần; sau đó người dùng đã duyệt nền và nhóm ghi nhận kiểm tra chéo nhãn; trạng thái hiện tại được kiểm chứng ngày 30/10/2026. Các ảnh hiệu chuẩn độc lập thử có người nên bị loại. Chưa dùng khung test làm nền thay thế.

Nhãn XML ban đầu được lọc sơ bộ theo tâm box trong ROI đã chốt; bộ chính hiện đã được người rà soát và kiểm tra chéo, không dùng XML tự động làm xác nhận. Không thu nhỏ ROI sau khi xem sai số test.

## 4. Khảo sát dev tuần 1

### Baseline trên 76 nhãn dev đã kiểm tra chéo

Run: outputs/experiments/20261029T171005Z_8392ad33ef. Kiểm chứng lại ngày 30/10/2026; config snapshot khớp configs/baseline.toml, số nhãn khớp counts.csv hiện tại.

| Phạm vi | N | MAE | RMSE | Đếm đúng | Bias |
|---|---:|---:|---:|---:|---:|
| Toàn bộ dev | 76 | 1.000 | 1.348 | 38.16% | -0.947 |
| Scene_01 | 26 | 0.846 | 1.209 | 42.31% | -0.692 |
| Scene_02 | 50 | 1.080 | 1.414 | 36.00% | -1.080 |

Các chỉ số cho biết sai lệch so với nhãn, chưa xác định lỗi phát sinh ở segmentation, morphology hay lọc vùng; cần xem ảnh và mask dev trước khi chọn cải tiến. Chưa có bù sáng giữa ảnh và nền.

### Khảo sát ban đầu ngày 28/10

**Bảng 7 biến thể dưới đây chỉ dựa trên cùng 26 nhãn cũ, không phải khảo sát đầy đủ 76 dev hay 75 test.** Chưa chạy lại toàn bộ các biến thể sau kiểm tra chéo nhãn, chưa chọn cấu hình cuối.

| Biến thể | MAE | RMSE | Đếm đúng | Bias |
|---|---:|---:|---:|---:|
| baseline | 0.846 | 1.209 | 42.3% | -0.692 |
| gamma | 0.885 | 1.286 | 42.3% | -0.731 |
| percentile | 1.077 | 1.441 | 30.8% | -0.692 |
| median | 0.769 | 1.074 | 42.3% | -0.615 |
| fixed_threshold | 1.423 | 1.808 | 26.9% | -1.269 |
| no_morphology | 0.923 | 1.240 | 34.6% | -0.615 |
| watershed | 1.115 | 1.532 | 34.6% | 0.192 |

Median cải thiện sơ bộ trên tập nhỏ; chưa thể kết luận tốt hơn cho nhiều cảnh. Watershed chưa cải thiện ở đây, giữ mặc định tắt và chỉ dùng minh họa/thử nghiệm. Trên hai hình tròn tổng hợp tiếp xúc, kiểm tra xác nhận nhánh tách có thể trả hai vùng thay vì một; đó không phải bằng chứng tách người thực tế luôn đúng.

Bảng thời gian lưu tại outputs/benchmarks/20261028T040847Z_83a8849403/timings.csv. Mỗi ảnh/backend có một warm-up và median 3 lần; chỉ tính preprocessing/segmentation/regions, không I/O, thống kê histogram hoặc render. Bản NumPy dùng diện tích pixel còn OpenCV dùng diện tích contour; đây không phải hai implementation hoàn toàn đồng nhất mọi bước.

## 5. Cách chạy ngay

```bash
cd /home/nth24075/Documents/IMG_processing/object_counting_gd1_2/oc2
uv sync --locked
uv run python app.py
```

Đếm ảnh mới bằng CLI:

```bash
uv run python predict.py --scene scene_02 --image data/week1/frames/scene_02/f00180.png
uv run python run_eval.py --config configs/baseline.toml --split dev
uv run python -m unittest discover -s tests -v
```

Rà soát nhãn và nền (thay Ten_that bằng tên người thực hiện):

```bash
uv run python review_labels.py --scene scene_02 --reviewer "Ten_that"
uv run python review_labels.py --scene scene_03 --reviewer "Ten_that"
uv run python review_labels.py --scene scene_01 --reviewer "Ten_that"
uv run python review_background.py --scene scene_02 --reviewer "Ten_that"
uv run python review_background.py --scene scene_03 --reviewer "Ten_that"
uv run python validate_dataset.py
```

Mỗi ảnh cần người khác rà soát lại; trạng thái peer_reviewed chỉ tạo khi hai người khác nhau xác nhận cùng số. Sửa số đếm phải duyệt chéo lại. Không dùng hai tên cho một người. Công cụ nhãn không chạy thuật toán đếm trên test.

## 6. Checkpoint đã đạt và phần còn mở

- Dữ liệu pass: 76 dev + 75 test, tất cả 151 ảnh chính peer_reviewed; tên hai người duyệt được ghi đầy đủ và khác nhau trong hồ sơ mỗi ảnh. Việc kiểm tra tên không xác minh danh tính ngoài đời.
- Nền scene_02 và scene_03 reviewed; scene_01 giữ legacy_offline_median. 27 holdout lịch sử vẫn human_labeled và ngoài bộ chính.
- Đã chạy lại 16 unittest thành công; chưa có final.toml hoặc test_freeze.json thật, test_evaluation_run=false.
- Xác nhận giảng viên về phạm vi người+cần nền, nguyên văn 5 bước/bài báo, hạn nộp và phụ lục vẫn chưa được cung cấp.

Bước tiếp theo là chẩn đoán lỗi trên dev, thử bù sáng hoặc sửa segmentation theo bằng chứng, đánh giá lại các biến thể trên cùng 76 ảnh đã kiểm chứng và chọn final theo dev. Chưa chạy đánh giá test. Nhãn mask nâng cao vẫn là phần tùy chọn, chưa thực hiện.

## 7. Tệp bàn giao trong dự án

README.md hướng dẫn cài/chạy; tasks/todo.md ghi checkpoint thật; docs/annotation_guide.md và decisions.md; reports/final_report/methods_week1.md là nháp phương pháp; reports/tables/week1_variants.csv và week1_timings.csv là bằng chứng dev. Mã và dữ liệu làm việc nằm trong oc2, không phải chỉ có tài liệu mô tả.
