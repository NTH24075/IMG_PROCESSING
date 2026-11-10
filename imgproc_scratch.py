"""GIAI ĐOẠN 2 (bản TỰ CÀI ĐẶT bằng NumPy, không dùng OpenCV): chứng minh hiểu bài giảng.
Mỗi hàm ghi rõ bài giảng tương ứng. Dùng để (1) viết phần 'cài đặt' của báo cáo,
(2) đối chiếu với bản OpenCV trong run_baseline.py.
"""
import numpy as np
from collections import deque
from numpy.lib.stride_tricks import sliding_window_view

def to_gray(bgr):                                    # chuyển ảnh xám [L2]
    b, g, r = [bgr[..., i].astype(np.float32) for i in range(3)]
    return np.round(0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

def histogram(gray):                                 # histogram 256 mức [L2]
    return np.bincount(gray.ravel(), minlength=256)

def otsu_threshold(gray):                            # chọn ngưỡng từ histogram chuẩn hóa p(i)=h(i)/MN [L2-3]
    p = histogram(gray) / gray.size
    w0 = np.cumsum(p); mu = np.cumsum(p * np.arange(256)); mu_t = mu[-1]
    var_b = (mu_t * w0 - mu) ** 2 / (w0 * (1 - w0) + 1e-12)   # phương sai giữa hai lớp
    return int(np.argmax(var_b))

def convolve2d(img, kernel):                         # tích chập/tương quan 2D, biên phản xạ [L3-4]
    kh, kw = kernel.shape; ph, pw = kh // 2, kw // 2
    pad = np.pad(img.astype(np.float32), ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros(img.shape, np.float32)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * pad[i:i + img.shape[0], j:j + img.shape[1]]
    return out

def gaussian_kernel_1d(k):
    sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8          # xấp xỉ sigma; OpenCV dùng kernel riêng ở một số k nhỏ
    x = np.arange(k) - k // 2
    g = np.exp(-x ** 2 / (2 * sigma ** 2)); return (g / g.sum()).astype(np.float32)

def gaussian_blur(img, k):                           # Gaussian tách được: 2 lượt 1D [L4]
    if k <= 1: return img
    g = gaussian_kernel_1d(k)
    out = convolve2d(convolve2d(img, g[None, :]), g[:, None])
    return np.clip(np.round(out), 0, 255).astype(np.uint8)

def abs_diff(a, b):                                  # hiệu tuyệt đối hai ảnh [L3]
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)

def disk(k):                                         # phần tử cấu trúc hình đĩa [L6]
    r = k // 2; y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r + r).astype(np.uint8)

def _win(img, se, pad_val):
    kh, kw = se.shape
    p = np.pad(img, ((kh // 2,) * 2, (kw // 2,) * 2), constant_values=pad_val)
    return sliding_window_view(p, (kh, kw))[..., se.astype(bool)]

def erode(m, se):  return _win(m, se, 255).min(-1)   # phép co [L6]
def dilate(m, se): return _win(m, se, 0).max(-1)     # phép giãn [L6]
def opening(m, se): return dilate(erode(m, se), se)  # [L7]
def closing(m, se): return erode(dilate(m, se), se)  # [L7]

def boundary(m, se3):                                # biên trong: A - (A erosion B) [L6]
    return (m > 0) & ~(erode(m, se3) > 0)

def label_components(mask, connectivity=8):          # gán nhãn bằng flood fill BFS [L7]
    labels = np.zeros(mask.shape, np.int32); H, W = mask.shape
    nb = [(-1, 0), (1, 0), (0, -1), (0, 1)] + ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if connectivity == 8 else [])
    comps, nxt = [], 1
    for y, x in np.argwhere(mask > 0):
        if labels[y, x]: continue
        q = deque([(y, x)]); labels[y, x] = nxt; pix = []
        while q:
            cy, cx = q.popleft(); pix.append((cy, cx))
            for dy, dx in nb:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = nxt; q.append((ny, nx))
        pix = np.array(pix)
        comps.append(dict(label=nxt, area=len(pix), bbox=(pix[:, 1].min(), pix[:, 0].min(), pix[:, 1].max(), pix[:, 0].max()),
                          centroid=tuple(pix.mean(0)[::-1])))
        nxt += 1
    return labels, comps

def count_scratch(img_bgr, bg_bgr, blur=5, thresh=0, min_thresh=20, se_open=None, se_close=None, min_area=150, max_area=30000):
    g = gaussian_blur(to_gray(img_bgr), blur); b = gaussian_blur(to_gray(bg_bgr), blur)
    d = abs_diff(g, b)
    t = thresh if thresh > 0 else max(otsu_threshold(d), min_thresh)
    raw = ((d > t) * 255).astype(np.uint8); m = raw
    if se_open is not None: m = opening(m, se_open)
    if se_close is not None: m = closing(m, se_close)
    labels, comps = label_components(m)
    keep = [c for c in comps if min_area <= c["area"] <= max_area]
    return len(keep), dict(gray=g, diff=d, thr=t, mask_raw=raw, mask=m, labels=labels, comps=keep)
