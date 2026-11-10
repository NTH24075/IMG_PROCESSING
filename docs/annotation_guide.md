# Gán nhãn và rà soát — tuần 1

## Bộ chính

- scene_01: 26 dev từ vtest, nhãn người dùng đã có. Không đổi file nhãn cũ.
- scene_02: 50 dev Meet_Crowd; số XML là nháp.
- scene_03: 75 test WalkByShop1cor; chỉ người phụ trách nhãn được rà soát, không dùng để tối ưu pipeline.
- 27 holdout lịch sử ngoài bộ chính, mặc định không xuất hiện trong review_labels.

## Quy tắc

Đếm người nhìn thấy khoảng ít nhất 50% cơ thể và nhận ra rõ; người cắt mép chỉ đếm nếu vẫn đạt quy tắc. Đếm cả người đứng yên đủ điều kiện. Nhãn XML có thể bỏ sót người đứng yên, chứa đối tượng ngoài ROI hoặc lệch frame MPEG; không chỉ đếm số box/ID. Không đếm grouplist thành một người nữa.

scene_01/02 dùng toàn ảnh. scene_03 có **ROI đã được chủ dự án chốt ngày 28/10/2026** x=0..383, y=65..287 (vùng hành lang gần), tách vùng xa có người đứng/di chuyển trong nền khởi tạo và không đủ nền trống. Ngoài ROI được tô tối trong công cụ; không đếm người ở phần tối. Rà soát nhãn và kiểm tra ảnh nền theo ROI đã chốt; việc chốt ROI không tự duyệt nhãn hoặc ảnh nền. Quyết định này dựa trên việc kiểm tra nền/warmup, chưa xem sai số test.

Quy tắc biên ROI: điểm đại diện tâm phần cơ thể nhìn thấy phải nằm trong ROI, đồng thời đạt điều kiện khoảng 50% cơ thể nhìn thấy. XML dùng tâm box để tạo đề xuất ban đầu, chưa thay cho xác nhận bằng mắt. Nếu đổi ROI phải cập nhật nhãn, manifest/cấu hình và kiểm tra lại; không thu nhỏ ROI sau khi xem kết quả để che lỗi.

## Hai lượt duyệt

Chạy từ thư mục oc2, thay tên minh họa bằng tên thật người đang rà soát:

```bash
uv run python review_labels.py --scene scene_02 --reviewer "Nguoi_1"
uv run python review_labels.py --scene scene_02 --reviewer "Nguoi_2"
```

Làm tương tự scene_01 (26 ảnh) và scene_03 (75 ảnh). Không dùng một người với hai tên để tạo duyệt giả.

- Đếm bằng mắt trước khi xác nhận. Ô số đếm đang hiển thị đề xuất; không bấm duyệt hàng loạt.
- Bật box XML chỉ để kiểm tra căn chỉnh; box không phải dự đoán của pipeline.
- `Lưu duyệt + tiếp` xác nhận rõ ràng số đếm. Enter có cùng chức năng.
- Ctrl+Left/Ctrl+Right di chuyển; bỏ qua không tự duyệt.
- Lượt đầu tạo reviewed. Người khác xác nhận cùng số tạo peer_reviewed.
- Nếu đổi số thì hủy duyệt chéo cũ và cần người khác kiểm tra lại.
- Mỗi lần lưu có bản CSV trước đó và review_log.jsonl trong data/week1/annotations/.

Trường hợp mơ hồ: giữ draft/bỏ qua rồi cùng thảo luận, không sửa để khớp dự đoán. Không chạy đếm trên test để gợi ý nhãn.

## Kiểm tra nền riêng

```bash
uv run python review_background.py --scene scene_02 --reviewer "Ten_that"
uv run python review_background.py --scene scene_03 --reviewer "Ten_that"
```

Chỉ bấm xác nhận khi nền không còn người/bóng người giả trong ROI. Kiểm tra nền và gán nhãn là hai việc khác nhau; nguồn đã tải không có nghĩa nền đã sạch. Nếu chưa đạt, giữ trạng thái chưa duyệt và chọn ảnh nền được cung cấp/hiệu chuẩn hợp lệ, không lấy từ phần test đang đánh giá.

## Kiểm tra tiến độ

```bash
uv run python validate_dataset.py
```

Ảnh XML chưa rà soát mặc định không được dùng tính chỉ số chính thức. --include-draft của evaluator chỉ dành cho dev và kết quả phải ghi là tạm thời; không áp dụng test. Freeze sẽ từ chối nếu nhãn chính chưa peer_reviewed hoặc nền chưa được duyệt.
