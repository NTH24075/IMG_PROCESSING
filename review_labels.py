"""Review multi-scene labels. No predictions are computed, including on test."""
import argparse
import base64
import json
import tkinter as tk
from tkinter import ttk
import cv2
from object_counting.io import ROOT, read_csv, read_image, resolve, load_scene
from object_counting.annotation import review_label


class LabelReviewApp:
    def __init__(self,root,scene,reviewer,include_legacy=False):
        self.root=root; self.reviewer=reviewer
        self.rows=[]
        for split in (('dev','test','legacy_holdout') if include_legacy else ('dev','test')):
            self.rows += [r for r in read_csv(ROOT/'data/week1/manifests'/f'{split}.csv') if r['scene_id']==scene]
        if not self.rows:
            raise ValueError('Không có ảnh của cảnh '+scene)
        self.index=0; self.count=tk.StringVar(); self.boxes=tk.BooleanVar(value=False)
        self.candidates=json.loads((ROOT/'data/week1/annotations/xml_candidates.json').read_text())
        root.title('Rà soát nhãn · '+scene); root.geometry('1000x780'); root.minsize(760,600)
        frame=ttk.Frame(root,padding=16); frame.pack(fill='both',expand=True)
        ttk.Label(frame,text='Rà soát số người — '+scene,font=('Sans',18,'bold')).pack(anchor='w')
        ttk.Label(frame,text='Đếm bằng mắt theo quy tắc nhóm. Nhãn XML là nháp; công cụ này không chạy mô hình.').pack(anchor='w',pady=8)
        self.description=ttk.Label(frame,text='',wraplength=950); self.description.pack(anchor='w')
        self.preview=ttk.Label(frame,anchor='center'); self.preview.pack(fill='both',expand=True,pady=10)
        bar=ttk.Frame(frame); bar.pack(fill='x')
        ttk.Label(bar,text='Số người').pack(side='left')
        ttk.Spinbox(bar,from_=0,to=999,textvariable=self.count,width=6).pack(side='left',padx=10)
        ttk.Button(bar,text='Ảnh trước',command=lambda:self.move(-1)).pack(side='left',padx=5)
        ttk.Button(bar,text='Lưu duyệt + tiếp',command=self.save).pack(side='left',padx=5)
        ttk.Button(bar,text='Bỏ qua / ảnh sau',command=lambda:self.move(1)).pack(side='left',padx=5)
        ttk.Checkbutton(frame,text='Hiện box XML để kiểm tra căn chỉnh (không phải dự đoán)',variable=self.boxes,command=self.show).pack(anchor='w',pady=10)
        self.status=ttk.Label(frame,text='Người duyệt: '+reviewer,wraplength=950); self.status.pack(anchor='w')
        root.bind('<Return>',lambda _:self.save()); root.bind('<Control-Left>',lambda _:self.move(-1)); root.bind('<Control-Right>',lambda _:self.move(1))
        self.photo=None; self.show()

    def show(self):
        row=self.rows[self.index]
        labels={r['image_id']:r for r in read_csv(ROOT/'data/week1/annotations/counts.csv')}
        label=labels[row['image_id']]; self.count.set(label['count'])
        self.description.config(text=f"{self.index+1}/{len(self.rows)} · {row['image_id']} · {row['split']} · {label['review_status']} · người trước: {label['reviewer'] or 'chưa có'}")
        image=read_image(resolve(row['image']))
        _,_,roi=load_scene(row['scene_id'])
        if roi is not None:
            image[roi==0]=(image[roi==0].astype(float)*.25).astype('uint8')
            contours,_=cv2.findContours(roi,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image,contours,-1,(0,220,255),1)
        if self.boxes.get():
            for obj in self.candidates.get(row['image_id'],[]):
                x,y,w,h=obj['xc'],obj['yc'],obj['w'],obj['h']
                cv2.rectangle(image,(int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(0,200,255),1)
        scale=min(920/image.shape[1],520/image.shape[0])
        image=cv2.resize(image,(int(image.shape[1]*scale),int(image.shape[0]*scale)))
        self.photo=tk.PhotoImage(data=base64.b64encode(cv2.imencode('.png',image)[1].tobytes()))
        self.preview.config(image=self.photo)

    def move(self,delta):
        self.index=max(0,min(len(self.rows)-1,self.index+delta)); self.show()

    def save(self):
        try:
            value=int(self.count.get())
            row=review_label(self.rows[self.index]['image_id'],value,self.reviewer)
            self.status.config(text='Đã lưu: '+row['review_status']+' · người duyệt: '+self.reviewer)
            if self.index<len(self.rows)-1:
                self.move(1)
            else:
                self.show(); self.status.config(text='Đã đến ảnh cuối. Kiểm tra còn nhãn draft hoặc chưa có người duyệt thứ hai.')
        except (ValueError,OSError) as error:
            self.status.config(text=str(error))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',choices=['scene_01','scene_02','scene_03'],required=True)
    p.add_argument('--reviewer',required=True,help='Tên người thật đang rà soát, không dùng chung một tên cho hai người')
    p.add_argument('--include-legacy',action='store_true',help='Chỉ khi muốn rà soát thêm holdout lịch sử, ngoài bộ chính')
    a=p.parse_args(); root=tk.Tk()
    try:
        LabelReviewApp(root,a.scene,a.reviewer,a.include_legacy)
    except (ValueError,OSError) as error:
        root.destroy(); p.exit(2,str(error)+'\n')
    root.mainloop()
