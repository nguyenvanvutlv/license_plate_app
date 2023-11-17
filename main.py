import tkinter as tk
from tkinter import Canvas
from utils import config
from PIL import ImageTk, Image
import cv2

from utils.opencv import (
    camera,
    draw,
    fps
)


class App(tk.Tk):
    def __init__(self, config, model_detection, ocr):
        super().__init__()
        self.config = config
        self.model_detection = model_detection
        self.ocr = ocr
        self.webcam = camera.Camera(self.config.input)
        # self.webcam = cv2.VideoCapture(self.config.input)
        self._fps = fps.FPS()

        self.geometry(f"{self.config.width}x{self.config.height}")
        self.title(self.config.title)

        logo_image = Image.open(self.config.logo)
        logo_photo = ImageTk.PhotoImage(logo_image)
        self.iconphoto(False, logo_photo)
        self.configure(bg=self.config.background)

        self.canvas = Canvas(
            self,
            bg=self.config.background,
            height=720,
            width=1280,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        self.canvas.place(x=0, y=0)
        showimg = ImageTk.PhotoImage(Image.open(self.config.showimg))
        self.canvas.showimg = showimg  # Giữ tham chiếu đến hình ảnh
        self.webcam_image = self.canvas.create_image(
            388.0,
            232.0,
            image=showimg
        )

        lp_img = ImageTk.PhotoImage(Image.open(self.config.lp_img))
        self.canvas.lp_img = lp_img  # Giữ tham chiếu đến hình ảnh
        self.licence_img = self.canvas.create_image(
            992.0,
            343.0,
            image=lp_img
        )

        opencamera = ImageTk.PhotoImage(Image.open(self.config.opencamera))
        self.canvas.opencamera = opencamera
        opencamera_button = tk.Button(
            self,
            image=opencamera,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.__opencamera(),
            relief="flat"
        )
        opencamera_button.place(
            x=210.0,
            y=472.0,
            width=147.0,
            height=42.0
        )

        close_camera = ImageTk.PhotoImage(Image.open(self.config.close_camera))
        self.canvas.close_camera = close_camera
        close_camera_button = tk.Button(
            self,
            image=close_camera,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.__close_camera(),
            relief="flat"
        )
        close_camera_button.place(
            x=465.0,
            y=472.0,
            width=147.0,
            height=42.0
        )

        self.canvas.create_text(
            883.0,
            34.0,
            anchor="nw",
            text="FPS",
            fill="#000000",
            font=("Inter Bold", 24 * -1)
        )

        self.show_fps = self.canvas.create_text(
            1050.0,
            34.0,
            anchor="nw",
            text="0",
            fill="#000000",
            font=("Inter Bold", 24 * -1)
        )

        self.canvas.create_text(
            883.0,
            120.0,
            anchor="nw",
            text="CONF",
            fill="#000000",
            font=("Inter Bold", 24 * -1)
        )

        self.show_conf = self.canvas.create_text(
            1050.0,
            120.0,
            anchor="nw",
            text="0",
            fill="#000000",
            font=("Inter Bold", 24 * -1)
        )

        self.canvas.create_text(
            883.0,
            198.0,
            anchor="nw",
            text="LICENCE",
            fill="#000000",
            font=("Inter Bold", 24 * -1)
        )

        self.show_licence = self.canvas.create_text(
            1050.0,
            198.0,
            anchor="nw",
            text="0",
            fill="#000000",
            font=("Inter Bold", 24 * -1)
        )

        self.canvas.create_rectangle(
            778.0,
            -1.0,
            781.0,
            720.0,
            fill="#000000",
            outline="")

    def __opencamera(self):
        self.webcam.start()
        self.__run__()

    def __close_camera(self):
        self.webcam.stop()
        self.webcam.release()
        self.webcam.restart()

    def __run__(self):
        self._fps.start()
        frame = self.webcam.read()
        if frame is not None:
            frame = self.inference(frame)

            self._fps.stop()
            self._fps.update()
            self.canvas.itemconfig(
                self.show_fps, text=f"{round(self._fps.fps(), 2)}")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            showimg = ImageTk.PhotoImage(Image.fromarray(frame).resize(
                (self.config.w_show,  self.config.h_show)))
            self.canvas.showimg = showimg
            self.canvas.itemconfig(self.webcam_image, image=showimg)
            self.after(10, self.__run__)

    def inference(self, frame):
        xyxys, confs = self.model_detection.predict(
            frame, self.config.confidence)
        frame = self.model_detection.draw(frame, xyxys)
        self.canvas.itemconfig(self.show_conf, text=f"{int(confs * 100)}")

        for id, xyxy in enumerate(xyxys):
            croped = frame.copy()[xyxy[1]: xyxy[3], xyxy[0]: xyxy[2]]

            ratio = max(xyxy[3] - xyxy[1], xyxy[2] - xyxy[0]) / \
                min(xyxy[3] - xyxy[1], xyxy[2] - xyxy[0])
            frame = draw.putText(frame, f"{round(ratio, 2)}", (20, 20))

            if ratio <= 2:
                # licence plate 2 line
                pass
            else:
                # licence plate 1 line
                text_licence = self.ocr.run(croped)
                self.canvas.itemconfig(self.show_licence, text=text_licence)
        return frame


if __name__ == "__main__":
    args, model_detection, ocr = config.make_parser()
    app = App(args, model_detection, ocr)
    app.mainloop()
