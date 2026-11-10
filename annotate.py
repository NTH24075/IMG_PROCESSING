"""GIAI ĐOẠN 1b: gán nhãn đếm bằng cách CLICK vào từng đối tượng (người).
Chuột trái: thêm điểm | Chuột phải hoặc phím u: xóa điểm gần nhất/cuối | Space hoặc n: lưu + ảnh kế
b: ảnh trước | q: lưu + thoát.   Mặc định chỉ hiện ảnh chưa gán nhãn (thêm --all để xem lại tất cả).
Quy ước đếm (GHI VÀO BÁO CÁO): đếm người nhìn thấy ít nhất ~50% cơ thể; người bị cắt ở mép ảnh vẫn đếm nếu thấy rõ.
"""
import argparse, json, os, cv2, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--frames", default="data/frames")
ap.add_argument("--gt", default="data/ground_truth.csv")
ap.add_argument("--points", default="data/annotations.json")
ap.add_argument("--all", action="store_true")
a = ap.parse_args()

df = pd.read_csv(a.gt); df["count"] = df["count"].astype("Int64")
pts_db = json.load(open(a.points)) if os.path.exists(a.points) else {}
todo = [i for i in range(len(df)) if a.all or pd.isna(df.loc[i, "count"])]
if not todo: print("Đã gán nhãn xong tất cả."); raise SystemExit

def save():
    for name, pts in pts_db.items():
        df.loc[df.image == name, "count"] = len(pts)
    df.to_csv(a.gt, index=False); json.dump(pts_db, open(a.points, "w"))

pos, pts = 0, []
def on_mouse(ev, x, y, *_):
    if ev == cv2.EVENT_LBUTTONDOWN: pts.append([x, y])
    elif ev == cv2.EVENT_RBUTTONDOWN and pts:
        j = min(range(len(pts)), key=lambda k: (pts[k][0]-x)**2 + (pts[k][1]-y)**2); pts.pop(j)

cv2.namedWindow("annotate"); cv2.setMouseCallback("annotate", on_mouse)
while 0 <= pos < len(todo):
    name = df.loc[todo[pos], "image"]
    if not pts and name in pts_db: pts[:] = pts_db[name]
    img = cv2.imread(f"{a.frames}/{name}")
    while True:
        v = img.copy()
        for k, (x, y) in enumerate(pts):
            cv2.circle(v, (x, y), 6, (0, 0, 255), -1); cv2.putText(v, str(k+1), (x+8, y-8), 0, 0.5, (0, 255, 255), 2)
        cv2.putText(v, f"{name}  [{pos+1}/{len(todo)}]  count={len(pts)}", (10, 25), 0, 0.7, (0, 255, 0), 2)
        cv2.imshow("annotate", v); k = cv2.waitKey(30) & 0xFF
        if k == ord("u") and pts: pts.pop()
        elif k in (ord(" "), ord("n"), 13): pts_db[name] = [list(p) for p in pts]; pts.clear(); pos += 1; save(); break
        elif k == ord("b"): pts.clear(); pos = max(0, pos-1); break
        elif k == ord("q"): pts_db[name] = [list(p) for p in pts]; save(); raise SystemExit
save(); print("Xong.")
