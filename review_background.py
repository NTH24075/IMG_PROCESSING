"""Human visual confirmation of an already prepared background; no inference."""
import argparse
import base64
import json
import tkinter as tk
from tkinter import ttk
from datetime import datetime, timezone
from object_counting.io import ROOT, load_scene
import cv2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',required=True,choices=['scene_01','scene_02','scene_03'])
    parser.add_argument('--reviewer',required=True)
    args=parser.parse_args()
    if not args.reviewer.strip():parser.error('Cần tên người thật đang kiểm tra')
    scene,image,roi=load_scene(args.scene)
    if roi is not None:
        image[roi==0]=(image[roi==0].astype(float)*.25).astype('uint8')
        contours,_=cv2.findContours(roi,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image,contours,-1,(0,220,255),1)
    root=tk.Tk();root.title('Kiểm tra nền — '+args.scene)
    frame=ttk.Frame(root,padding=16);frame.pack(fill='both',expand=True)
    ttk.Label(frame,text='Chỉ xác nhận nếu không còn người/bóng người giả trong vùng cần đếm.',wraplength=850).pack(anchor='w',pady=8)
    ttk.Label(frame,text=scene['background_policy'],wraplength=850).pack(anchor='w',pady=8)
    scale=min(850/image.shape[1],580/image.shape[0])
    image=cv2.resize(image,(int(image.shape[1]*scale),int(image.shape[0]*scale)))
    photo=tk.PhotoImage(data=base64.b64encode(cv2.imencode('.png',image)[1].tobytes()))
    ttk.Label(frame,image=photo).pack()
    status=ttk.Label(frame,text='Nếu nền còn người: đóng cửa sổ, chưa duyệt và xử lý lại nền theo protocol.');status.pack(pady=8)
    def approve():
        path=ROOT/'configs/scenes'/f'{args.scene}.toml'
        scene['background_status']='reviewed';scene['background_reviewer']=args.reviewer.strip()
        if scene.get('roi_proposal_status'):
            scene['roi_proposal_status']='reviewed'
        scene['background_reviewed_at']=datetime.now(timezone.utc).isoformat()
        path.write_text('[scene]\n'+'\n'.join(f'{k} = {json.dumps(v,ensure_ascii=False)}' for k,v in scene.items())+'\n')
        with (ROOT/'docs/decisions.md').open('a',encoding='utf-8') as f:
            f.write(f"\n- Background review {scene['background_reviewed_at']}: {args.scene}, reviewer={args.reviewer.strip()}, status=reviewed.\n")
        status.config(text='Đã ghi nhận người kiểm tra: '+args.reviewer.strip())
        button.state(['disabled'])
    button=ttk.Button(frame,text='Tôi đã kiểm tra — nền đạt yêu cầu',command=approve);button.pack(pady=8)
    root.mainloop()


if __name__=='__main__':
    main()
