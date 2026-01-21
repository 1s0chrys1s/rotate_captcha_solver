
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from rotate.src.double_rotate_solver import double_rotate_identify


class VerificationApp:
    def __init__(self, master):
        self.master = master
        self.master.title("旋转验证程序")


        self.outer_image_path = None
        self.inner_image_path = None
        self.outer_image = None
        self.inner_image = None

        # --- GUI 元素 ---
        self.main_frame = tk.Frame(master)
        self.main_frame.pack(padx=10, pady=10)

        # 按钮框架
        self.btn_frame = tk.Frame(self.main_frame)
        self.btn_frame.pack(pady=5)
        tk.Button(self.btn_frame, text="选择外圈图片", command=self.load_outer_image).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="选择内圈图片", command=self.load_inner_image).pack(side=tk.LEFT, padx=5)

        self.btn_verify = tk.Button(self.main_frame, text="开始验证", command=self.verify_and_rotate, state=tk.DISABLED)
        self.btn_verify.pack(pady=10)

        # 图片显示框架
        self.image_frame = tk.Frame(self.main_frame)
        self.image_frame.pack()
        self.canvas_orig = tk.Canvas(self.image_frame, width=250, height=150, bg="lightgray")
        self.canvas_orig.pack(side=tk.LEFT, padx=10)
        self.canvas_orig.create_text(125, 75, text="原始图片", fill="darkgray")
        self.canvas_rotated = tk.Canvas(self.image_frame, width=250, height=150, bg="lightgray")
        self.canvas_rotated.pack(side=tk.LEFT, padx=10)
        self.canvas_rotated.create_text(125, 75, text="自动校正后", fill="darkgray")
        
        self.angle_label = tk.Label(self.main_frame, text="预测角度: --", font=("Arial", 12))
        self.angle_label.pack(pady=5)

    def load_outer_image(self):
        path = filedialog.askopenfilename(title="选择外圈图片", filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if path:
            self.outer_image_path = path
            self.outer_image = Image.open(path).convert("RGBA")
            self.display_original_images()
            self.check_buttons()

    def load_inner_image(self):
        path = filedialog.askopenfilename(title="选择内圈图片", filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if path:
            self.inner_image_path = path
            self.inner_image = Image.open(path).convert("RGBA")
            self.display_original_images()
            self.check_buttons()

    def check_buttons(self):
        if self.outer_image_path and self.inner_image_path:
            self.btn_verify.config(state=tk.NORMAL)

    def display_original_images(self):
        if self.outer_image and self.inner_image:
            composite = self.outer_image.copy()
            center_x = (self.outer_image.width - self.inner_image.width) // 2
            center_y = (self.outer_image.height - self.inner_image.height) // 2
            composite.paste(self.inner_image, (center_x, center_y), self.inner_image)
            
            self.orig_img_tk = ImageTk.PhotoImage(composite.resize((250, 150)))
            self.canvas_orig.delete("all")
            self.canvas_orig.create_image(125, 75, image=self.orig_img_tk)

    def rotate_and_display(self, inner_angle_cw, outer_angle_ccw):
        """根据 detect.py 的逻辑旋转两个图像并显示它们。"""
        # Pillow 的旋转方法中，正数代表逆时针旋转，所以：
        # - 内圈（顺时针）使用负角度。
        # - 外圈（逆时针）使用正角度。
        rotated_inner = self.inner_image.rotate(-inner_angle_cw, resample=Image.BICUBIC, expand=False)
        rotated_outer = self.outer_image.rotate(outer_angle_ccw, resample=Image.BICUBIC, expand=False)

        # 创建最终的合成图像
        composite_rotated = rotated_outer.copy()
        center_x = (rotated_outer.width - rotated_inner.width) // 2
        center_y = (rotated_outer.height - rotated_inner.height) // 2
        composite_rotated.paste(rotated_inner, (center_x, center_y), rotated_inner)

        self.rotated_img_tk = ImageTk.PhotoImage(composite_rotated.resize((250, 150)))
        self.canvas_rotated.delete("all")
        self.canvas_rotated.create_image(125, 75, image=self.rotated_img_tk)

    def verify_and_rotate(self):
        if not self.inner_image_path or not self.outer_image_path:
            return
        self.btn_verify.config(state=tk.DISABLED)

        try:
            match_data = double_rotate_identify(
                small_circle=self.inner_image_path,
                big_circle=self.outer_image_path,
                image_type=2,        # File paths
                speed_ratio=1,
                grayscale=False,
                standard_deviation=0,
                cut_pixel_value=0,
                check_pixel=10,
            )

            if match_data and match_data.similar > 0:
                inner_angle = match_data.total_rotate_angle
                outer_angle = inner_angle / 1
                # inner_angle = 211/2
                # outer_angle = 211/2
                
                print(f"预测结果: 内圈角度 (顺时针)={inner_angle:.2f}°, 外圈角度 (逆时针)={outer_angle:.2f}°")
                self.angle_label.config(text=f"预测角度: {inner_angle:.2f}°")
                
                # 显示最终旋转后的状态
                self.rotate_and_display(inner_angle, outer_angle)
            else:
                print("未能成功计算角度。")
                self.angle_label.config(text="预测角度: 失败")

        except Exception as e:
            print(f"调用 double_rotate_identify 时发生错误: {e}")
            self.angle_label.config(text="预测角度: 错误")
        finally:
            self.btn_verify.config(state=tk.NORMAL)


def main():
    root = tk.Tk()
    app = VerificationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
