# Nháp phương pháp — tuần 1

Đây là nội dung phương pháp đang có mã và kiểm tra, chưa phải báo cáo cuối. Abstract, kết quả test và đối chiếu bài báo để trống đến khi có nguồn/kết quả thật. Nội dung cần nhóm đọc, diễn đạt lại theo hiểu biết và bổ sung trích dẫn môn học.

## 1. Phạm vi và dữ liệu

Đếm người trong ảnh từ camera cố định, có nền tương ứng. Chế độ suy luận chỉ nhận ảnh, nền và cấu hình; nhãn chỉ xuất hiện trong đánh giá. Dữ liệu chính 151 ảnh: 26 vtest dev, 50 Meet_Crowd dev và 75 WalkByShop1cor test; toàn bộ đã có trạng thái peer_reviewed, kiểm chứng ngày 30/10/2026. 27 holdout vtest lịch sử ngoài đánh giá chính. Không huấn luyện học sâu.

Meet_Crowd được lấy mẫu theo chỉ số đọc thực tế từ 45 đến 486, bước 9, 25 FPS; nền từ mỗi 5 khung trong prefix [0,30). WalkByShop lấy từ 500 đến hết, bước 25; nền prefix [0,500), mỗi 5 khung. Các ảnh cùng cảnh cùng vai trò; không coi các ảnh gần nhau là quan sát thống kê độc lập. Metadata MPEG có số khung báo sai nên lấy theo vòng đọc, cần rà soát căn chỉnh XML.

Chủ dự án đã chốt ROI hành lang gần x=0..383, y=65..287 ngày 28/10/2026 do nền vùng xa còn người. Quyết định dựa trên nền/warmup, trước khi xem kết quả đếm test. Scene_03 chỉ đếm trong vùng này; nền hiện đã reviewed và nhãn đã peer_reviewed.

## 2. Biểu diễn ảnh

Ảnh màu BGR uint8 H×W×3 được giữ để hiển thị. Ảnh xám uint8 H×W dùng cho xử lý; mask dùng 0 cho nền và 255 cho foreground. Tọa độ hình học (x,y) khác chỉ số mảng [y,x]. Ảnh/nền/ROI phải cùng kích thước; không resize tùy ý giữa các bước.

## 3. Các kỹ thuật chính

### 3.1. Histogram và enhancement

Histogram h(i) đếm pixel mức xám i; p(i)=h(i)/(HW), CDF(k)=Σ(i<=k)p(i). Tính brightness bằng trung bình pixel, spread bằng P95−P5, saturation bằng tỷ lệ pixel 0/255. Các thống kê hỗ trợ mô tả, không phải bộ phân loại ảnh tốt/xấu tổng quát.

Gamma dùng LUT s=255(r/255)^gamma, làm tròn và clamp. Contrast stretching dùng percentile 2/98 của nền để tạo cùng một LUT áp dụng cả nền và ảnh. Nền hằng thì giữ nguyên để tránh chia 0. Không cân bằng riêng hai histogram rồi mặc định phép trừ vẫn nhất quán.

Nhánh chính không enhancement giữ làm đối chứng. Gamma .7 và stretching chỉ là cấu hình khảo sát ban đầu, không phải final đã chọn.

### 3.2. Filtering và nhiễu

Gaussian k=3 là baseline, so với median k=3. Có nhánh mean để khảo sát bổ sung. Bản NumPy thực hiện Gaussian tách được bằng hai phép tương quan 1D, biên phản xạ; kernel Gaussian nhỏ xấp xỉ có thể khác kernel đặc biệt của OpenCV, nên không đòi ảnh đầu ra khớp từng pixel.

Thử minh họa nhiễu Gaussian sigma=15 và muối tiêu fraction=.02, seed=42; không chèn các ảnh này sang split khác. Hình gồm ảnh nguồn, nhiễu, Gaussian, median; chưa coi ảnh đẹp hơn là đếm tốt hơn.

### 3.3. Edge detection

Một bộ minh họa dev dùng Gaussian 3×3, Sobel x/y và magnitude; Canny 50/150. Không xây một nhánh đếm từ biên hoặc khảo sát tham số riêng trong tuần 1. Biên đứt và texture khiến edge map không tự trở thành mask người.

### 3.4. Segmentation

D=|I−B| trên ảnh xám đã xử lý, dùng kiểu trung gian để không tràn uint8. Ngưỡng cố định hoặc Otsu; với Otsu baseline dùng sàn 20. Pixel foreground khi D>t. Otsu chọn t nhằm tối đa phương sai giữa hai lớp từ histogram. Cấu hình ngưỡng cố định 25 chỉ để so sánh.

Pipeline không dùng nhãn để đặt ngưỡng riêng từng ảnh. ROI áp vào mask trước và sau morphology để giới hạn vùng có thể được đếm.

### 3.5. Morphology, regions và contours

Opening=erosion rồi dilation, closing=dilation rồi erosion; structuring element ellipse mặc định 3×3. Opening có thể xóa chi tiết/người nhỏ; closing quá lớn có thể gộp người. Có nhánh bỏ morphology để làm đối chứng.

OpenCV dùng contour ngoài và diện tích contour trong [150,30000]. Đặc trưng lưu: bbox, aspect ratio, extent, solidity. Chưa thêm các ngưỡng hình dạng mới nếu không có chứng cứ dev. Bản tự cài dùng BFS liên thông 8 và số pixel thành phần; diện tích này khác contour area nên hai số đếm không buộc bằng nhau.

Nhánh tách thử nghiệm dùng distance transform, threshold core theo tỷ lệ, connected components tạo marker và watershed; contour theo từng marker được lọc diện tích. Có kiểm tra tổng hợp hai vùng tiếp xúc. Mặc định tắt vì khảo sát nhãn dev cũ chưa có cải thiện, không hứa khôi phục người bị che khuất hoàn toàn.

## 4. Đánh giá và kiểm chứng

MAE=mean(|p−g|), RMSE=sqrt(mean((p−g)^2)), Bias=mean(p−g), exact-count accuracy=mean(p=g). Nhãn 0 hợp lệ. MAPE_modified chia max(g,1) được ghi riêng, không gọi là MAPE chuẩn.

Evaluator dùng cùng pipeline với demo. Nhãn draft mặc định bị loại khỏi chỉ số nhưng vẫn báo số ảnh và độ phủ nhãn. IoU mask OpenCV–NumPy là độ đồng thuận hai implementation, không phải IoU ground truth. Ground truth mask nâng cao chưa thực hiện; đây là phần tùy chọn sau khi hoàn thành các mốc chính.

Chỉ số baseline mới vẫn khớp số cũ trên 26 dev: OpenCV MAE .846, RMSE 1.209, đếm đúng 42.3%; NumPy MAE .808, RMSE 1.056, đếm đúng 34.6%. Bảng 7 cấu hình ở reports/tables/week1_variants.csv chỉ dùng 26 nhãn này, không chốt phương án đa cảnh.

Có 16 kiểm tra tự động và thử GUI callback; kiểm tra ảnh tổng hợp, tính tay, dữ liệu rỗng/lỗi, ROI, label peer-review, baseline hồi quy và gate/freeze/audit giả lập. Chưa có test thật được đánh giá. Ngày 30/10/2026 đã kiểm chứng nhãn/nền và chạy lại 16 kiểm tra tự động thành công; xác nhận yêu cầu môn học còn mở.

## 5. Phần chưa hoàn tất

Toàn bộ 151 nhãn bộ chính đã peer_reviewed, nền scene_02/03 reviewed; checkpoint dữ liệu và phần mềm tuần 1 đạt. Baseline dev đủ 76 ảnh cho MAE=1.000, RMSE=1.348, exact-count Acc=38.16%, Bias=-0.947 (run 20261029T171005Z_8392ad33ef); các số đã được tính lại từ bảng dự đoán và đối chiếu nhãn hiện tại. Chưa xác định nguyên nhân sai số qua mask từng ảnh; chưa cài bù sáng hoặc khảo sát lại đầy đủ các biến thể trên 76 nhãn đã kiểm tra chéo. Bài báo/5 bước/ngày nộp chưa có nguyên văn, không tự tạo kết quả/đối chiếu. Chưa có final.toml hoặc test_freeze.json thật; báo cáo tuần 2 sẽ bổ sung kết quả sau đóng băng.

## Tài liệu tra cứu

- Bài giảng môn học: lecture02, lecture03, lab04, lecture05, lecture06, lecture07 do nhóm có sẵn.
- OpenCV Histogram Equalization: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
- OpenCV Thresholding: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
- OpenCV Contour Properties: https://docs.opencv.org/4.x/d1/d32/tutorial_py_contour_properties.html
- OpenCV Watershed: https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
- CAVIAR nguồn/điều kiện dùng: https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/CAVIARDATA1/
