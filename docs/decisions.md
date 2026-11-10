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
