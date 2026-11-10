"""Local Tk desktop demo. Uses predict.py without labels or extra dependencies."""
import argparse
import base64
import tkinter as tk
from tkinter import ttk, filedialog
import cv2
from predict import predict
from object_counting.io import load_scene


class CountingApp:
    def __init__(self, root, image='', scene='scene_01', config='configs/baseline.toml'):
        self.root=root; root.title('Object Counting · Phòng thí nghiệm ảnh'); root.geometry('1100x780')
        root.minsize(760,580)
        style=ttk.Style(); style.configure('TButton',padding=7); style.configure('TLabel',font=('Sans',11))
        frame=ttk.Frame(root,padding=20); frame.pack(fill='both',expand=True)
        ttk.Label(frame,text='Đếm người từ ảnh giám sát',font=('Sans',19,'bold')).pack(anchor='w')
        ttk.Label(frame,text='Ảnh mới không cần nhãn · Cần ảnh nền cùng góc camera').pack(anchor='w',pady=(4,15))
        controls=ttk.Frame(frame); controls.pack(fill='x')
        self.scene=tk.StringVar(value=scene); self.image=tk.StringVar(value=image); self.config=tk.StringVar(value=config)
        ttk.Label(controls,text='Cảnh camera').grid(row=0,column=0,sticky='w')
        ttk.Combobox(controls,textvariable=self.scene,values=['scene_01','scene_02','scene_03'],state='readonly',width=13).grid(row=1,column=0,sticky='w',padx=(0,10))
        ttk.Label(controls,text='Ảnh đầu vào').grid(row=0,column=1,sticky='w')
        ttk.Entry(controls,textvariable=self.image).grid(row=1,column=1,sticky='ew',padx=(0,10))
        ttk.Button(controls,text='Chọn ảnh…',command=self.choose).grid(row=1,column=2)
        controls.columnconfigure(1,weight=1)
        options=ttk.Frame(frame); options.pack(fill='x',pady=12)
        ttk.Label(options,text='Cấu hình').pack(side='left')
        ttk.Entry(options,textvariable=self.config,width=48).pack(side='left',padx=10)
        self.run_button=ttk.Button(options,text='Đếm ảnh',command=self.run); self.run_button.pack(side='left')
        self.result=ttk.Label(frame,text='Chưa có kết quả',font=('Sans',15,'bold')); self.result.pack(anchor='w',pady=5)
        self.status=ttk.Label(frame,text='Chọn đúng cảnh và ảnh. scene_03 là test, đang khóa trước freeze.',wraplength=1000)
        self.status.pack(anchor='w',pady=(0,10))
        self.preview=ttk.Label(frame,anchor='center'); self.preview.pack(fill='both',expand=True)
        self.saved=ttk.Label(frame,text='',wraplength=1000); self.saved.pack(anchor='w',pady=(10,0))
        root.bind('<Control-o>',lambda _:self.choose()); root.bind('<Control-Return>',lambda _:self.run())
        self.photo=None; self.last_output=None

    def choose(self):
        file=filedialog.askopenfilename(title='Chọn ảnh từ camera tương ứng',filetypes=[('Images','*.png *.jpg *.jpeg *.bmp'),('All files','*')])
        if file:
            self.image.set(file)

    def run(self):
        self.run_button.state(['disabled']); self.status.config(text='Đang xử lý…'); self.root.update_idletasks()
        try:
            if not self.image.get().strip():
                raise ValueError('Bạn cần chọn ảnh trước khi đếm.')
            result,path=predict(self.image.get(),self.scene.get(),self.config.get())
            image=cv2.imread(str(path/'panel.png'))
            max_width=max(700,self.root.winfo_width()-60)
            max_height=max(240,self.root.winfo_height()-290)
            scale=min(max_width/image.shape[1],max_height/image.shape[0],1)
            image=cv2.resize(image,(int(image.shape[1]*scale),int(image.shape[0]*scale)))
            self.photo=tk.PhotoImage(data=base64.b64encode(cv2.imencode('.png',image)[1].tobytes()))
            self.preview.config(image=self.photo)
            self.result.config(text=f'Dự đoán: {result.count} người · {result.timings_ms["pipeline"]:.1f} ms xử lý')
            scene,_,_=load_scene(self.scene.get())
            self.status.config(text='Hoàn tất. Trạng thái nền: '+scene.get('background_status','unknown')+' — số đếm là dự đoán, không phải nhãn thật.')
            self.saved.config(text='Kết quả đã lưu: '+str(path)); self.last_output=path
        except Exception as error:
            self.result.config(text='Không tạo dự đoán'); self.status.config(text=str(error)); self.saved.config(text='')
            self.preview.config(image=''); self.photo=None; self.last_output=None
        finally:
            self.run_button.state(['!disabled'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',default=''); parser.add_argument('--scene',default='scene_01')
    parser.add_argument('--config',default='configs/baseline.toml'); args=parser.parse_args()
    root=tk.Tk(); CountingApp(root,args.image,args.scene,args.config); root.mainloop()


if __name__=='__main__':
    main()
