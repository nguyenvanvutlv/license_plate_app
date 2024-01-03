

"""
this file get all parameters for system, load model detection, ocr
"""
import argparse
from utils.licence.detect import YOLOv8
from utils.ocr.detect import OCRLicence

# BUG
# import lisence.darknet as dn


def make_parser():
    parser = argparse.ArgumentParser("Automatic License Plate Recognition")

    tkinter_gui = parser.add_argument_group("Config window")
    tkinter_gui.add_argument("--width", type=int, default=1280)
    tkinter_gui.add_argument("--height", type=int, default=720)
    # (654, 375)
    tkinter_gui.add_argument("--h_show", type=int, default=375)
    tkinter_gui.add_argument("--w_show", type=int, default=654)
    tkinter_gui.add_argument(
        "--title", type=str, default="Automatic Licence Plate Recognition")
    tkinter_gui.add_argument(
        "--logo", type=str, default="assets/motorcycle.png")
    tkinter_gui.add_argument("--background", type=str, default="#FFFFFF")
    tkinter_gui.add_argument(
        "--showimg", type=str, default="assets/image_1.png")
    tkinter_gui.add_argument(
        "--lp_img", type=str, default="assets/lp.png")
    tkinter_gui.add_argument("--opencamera", type=str,
                             default="assets/button_1.png")
    tkinter_gui.add_argument("--close_camera", type=str,
                             default="assets/button_2.png")

    detection = parser.add_argument_group("Detection")
    detection.add_argument("--input", type=str or int,
                           default=0)
    detection.add_argument("--device", type=str, default="cuda:0")
    detection.add_argument(
        "--model", type=str, default="model/detect/yolov8m.pt")
    detection.add_argument("--confidence", type=float, default=0.7)

    recognition = parser.add_argument_group("Recognition")
    recognition.add_argument(
        "--weights", type=str, default="model/recognition/ocr.pth")
    recognition.add_argument("--ocr_threshold", type=float, default=0.7)
    args = parser.parse_args()

    model = YOLOv8(source=args.model, device=args.device)
    ocr = OCRLicence(weight=args.weights, device=args.device)

    return args, model, ocr
