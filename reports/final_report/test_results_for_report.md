# Kết quả kiểm thử và số liệu để điền báo cáo (21/11/2026)

Tất cả các lần chạy test dùng commit `8879dd4` có thẻ `freeze-test`, cây sạch. Mỗi phương pháp chạy test **một lần** và được ghi trong `outputs/test_runs.log`. Cảnh kiểm thử là WalkByShop1cor (scene_03), 75 ảnh, 266 người, ROI hành lang gần. C3 không có mô hình phối cảnh trên cảnh này nên dùng quy tắc 1 vùng = 1 người.

## Bảng 6 — dòng kiểm thử (75 ảnh)

| Phương pháp | MAE (KTC 95%) | RMSE | Bias | Đúng | ms/ảnh |
|---|---|---|---|---|---|
| Cơ sở | 1,160 (0,89–1,45) | 1,400 | −0,867 | 20,0% | 0,8 |
| HOG + SVM | 2,440 (2,08–2,83) | 2,859 | +2,440 | 8,0% | 741 |
| Đề xuất (không có C3) | 1,213 (0,89–1,53) | 1,465 | −0,893 | 20,0% | 5,1 |

- ∆MAE ghép cặp, Đề xuất − Cơ sở: **+0,053 (KTC −0,04 – 0,16)**. Không có khác biệt có ý nghĩa; đề xuất hơi kém hơn.
- ∆MAE ghép cặp, HOG − Đề xuất: **+1,227 (KTC 0,61 – 1,87)**.
- Phân loại lỗi trên tập test, theo thứ tự gộp / thừa / tách / giả / sót:
  - Cơ sở: 44 / 0 / 3 / 51 / 40.
  - Đề xuất: 49 / 0 / 4 / 51 / 38.
- Thời gian đo trên máy cloud (lần chạy này), không phải máy của nhóm. HOG chậm hơn so với lần đo trên dev ở máy nhóm.

## Độ bền trên cảnh kiểm thử (robustness_test, 75 ảnh × 17 mức suy giảm)

MAE lớn nhất trên các mức của từng kiểu suy giảm:

| | Ảnh gốc | Gamma | Tương phản | Khuếch đại | Nhiễu Gauss | Muối tiêu |
|---|---|---|---|---|---|---|
| Cơ sở | 1,16 | 2,13 | **7,96** (a=0,3) | 1,36 | 1,20 | 1,21 |
| Đề xuất | 1,21 | 1,27 | 1,25 | 1,24 | 1,23 | 1,17 |

Trên cảnh này C3 tắt, nên nhiễu muối tiêu không còn làm pipeline đề xuất đếm thừa như trên dev.

## Câu gợi ý để điền chỗ trống

- **Abstract:** "Trên 75 ảnh kiểm thử của cảnh chưa thấy, nơi C3 không có mô hình phối cảnh, MAE = 1,21 so với 1,16 của phương pháp cơ sở (chênh lệch +0,05, KTC −0,04–0,16) và 2,44 của HOG + SVM; với suy giảm tương phản, MAE xấu nhất của pipeline là 1,25, so với 7,96 của phương pháp cơ sở."
- **Kết luận:** "Trên cảnh kiểm thử, nơi C3 không chạy, pipeline không tốt hơn phương pháp cơ sở trên ảnh gốc (1,21 so với 1,16) nhưng giữ MAE không quá 1,27 dưới mọi suy giảm cường độ toàn cục, trong khi phương pháp cơ sở lên tới 7,96."
- **Mục 4.1, số lần chạy test:** mỗi phương pháp một lần (nhật ký có 5 cặp start/done: cơ sở, đề xuất, HOG, và hai biến thể của benchmark suy giảm).

## Sửa câu chữ trong báo cáo theo kết quả kiểm chứng code

1. **§3.4 C2:** "Với cận trên 0,85, MAE trở lại 1,000" là sai. Chỉ đổi cận trên thì MAE vẫn là 0,987. MAE 1,000 xuất hiện khi đổi thêm ngưỡng gradient từ 20 xuống 15. Viết lại thành: "Với cận trên 0,85 và ngưỡng gradient 15, MAE trở lại 1,000."
2. **Bảng 1:** con số 98,5–99,6% của Gauss là tỉ lệ pixel lệch không quá ±1; tỉ lệ trùng tuyệt đối khoảng 85%. Với "So khớp histogram", cột OpenCV thực chất là bản tham chiếu NumPy (np.interp), vì OpenCV không có hàm so khớp histogram.
3. **§3.3 Hough:** nên nói thêm rằng trước khi so với ngưỡng 35% chu vi, mỗi ô của bộ tích lũy được cộng với lân cận 3×3.
4. **§4.1:** demo (`predict.py`, `app.py`) nay mặc định dùng cấu hình cuối; `predict.py` tự nhận cảnh từ ảnh nền nếu không chỉ định tên cảnh.
5. **§3, Dữ liệu:** ảnh test bắt đầu từ chính khung 500 (f00500), không phải "sau khung 500".

## Số dev đã tái lập từ code (khớp báo cáo)

- Bảng 5 và 7, gồm cả cột "Thừa": khớp hoàn toàn.
- Bảng 6, dòng kiểm định chéo: `run_c3_cv.py`, run `20261121T033819Z_9144319d2f`. K=5 cho MAE 0,868 (0,49–1,26), RMSE 1,203, Bias −0,421, đúng 42,1%.
- KTC theo độ dài khối 1/3/5/10: `run_bootstrap_sweep.py`; kết quả 0,43–0,78 / 0,33–0,88 / 0,26–0,97 / 0,24–1,07.
