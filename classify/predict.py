# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 classification inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python classify/predict.py --weights yolov5s-cls.pt --source 0                               # webcam
                                                                   img.jpg                         # image
                                                                   vid.mp4                         # video
                                                                   screen                          # screenshot
                                                                   path/                           # directory
                                                                   list.txt                        # list of images
                                                                   list.streams                    # list of streams
                                                                   'path/*.jpg'                    # glob
                                                                   'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                                   'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python classify/predict.py --weights yolov5s-cls.pt                 # PyTorch
                                           yolov5s-cls.torchscript        # TorchScript
                                           yolov5s-cls.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                           yolov5s-cls_openvino_model     # OpenVINO
                                           yolov5s-cls.engine             # TensorRT
                                           yolov5s-cls.mlmodel            # CoreML (macOS-only)
                                           yolov5s-cls_saved_model        # TensorFlow SavedModel
                                           yolov5s-cls.pb                 # TensorFlow GraphDef
                                           yolov5s-cls.tflite             # TensorFlow Lite
                                           yolov5s-cls_edgetpu.tflite     # TensorFlow Edge TPU
                                           yolov5s-cls_paddle_model       # PaddlePaddle
"""

import argparse
import math
import os
import platform
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import datetime

import torch
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation


# from diplom_wrapper.plot_draw import BarPlotClass


def get_root():
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]  # YOLOv5 root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    return Path(os.path.relpath(ROOT, Path.cwd()))  # relative


ROOT = get_root()

from models.common import DetectMultiBackend
from utils.augmentations import classify_transforms
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, print_args, strip_optimizer)
from utils.plots import Annotator
from utils.torch_utils import select_device, smart_inference_mode

global_radar_chart_name = 'Эмоциональная гексограмма за всё время'
last_5_min_radar_chart_name = 'Эмоциональная гексограмма за последние 5 минут'


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s-cls.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(224, 224),  # inference size (height, width)
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        nosave=False,  # do not save images/videos
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/predict-cls',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    def average_in_list(lst):
        return sum(lst) / len(lst)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Create plots
    matplotlib.use("TkAgg")

    lines_chart_history_size = 1000

    lines_data_y = [[0] * lines_chart_history_size for i in range(len(names))]
    lines_dirty_data_y = [[0] * lines_chart_history_size for i in range(len(names))]

    lines_data_x = [*range(-lines_chart_history_size, 0, 1)]
    sum_time_global = {names[i]: 1.0 for i in range(0, len(names), 1)}
    emotions_map_time_list = [{names[i]: 0 for i in range(0, len(names), 1)}]

    # Current time
    timer = datetime.datetime
    start = timer.now()

    def bar_chart_upd(prob):
        data_array = prob.tolist()
        bar_y = np.array(data_array)
        for rect, h in zip(barPlot, bar_y):
            rect.set_height(h)

    def pie_chart_upd():
        pie_values = list(sum_time_global.values())
        pie_keys = list(sum_time_global.keys())
        axPie.clear()
        axPie.pie(np.array(pie_values), labels=pie_keys)

    def lines_chart_upd(prob):
        frame_time = (timer.now() - start).total_seconds()
        data_array = prob.tolist()
        if frame_time - lines_data_x[-1] >= 1:
            lines_data_x.append(math.trunc(frame_time))
            for i in range(len(names)):
                mean_value = average_in_list(lines_dirty_data_y[i])
                lines_data_y[i].append(mean_value)
                lines_dirty_data_y[i] = [mean_value]
                axs2[i].set_yticklabels([math.trunc(mean_value * 100)])
                new_ls = lines_data_y[i][-lines_chart_history_size:]
                emotion_chart_list[i].set_ydata(new_ls)
                emotion_chart_list[i].set_xdata(lines_data_x[-lines_chart_history_size:])
        else:
            for i in range(len(names)):
                lines_dirty_data_y[i].append(data_array[i])

    def radar_chart_upd():
        global_radar_chart_upd()
        last_5_min_radar_chart_upd()

    def global_radar_chart_upd():
        radar_values = list(sum_time_global.values())
        radar_values_percent = list(map(lambda x: x / sum(radar_values), radar_values))
        radar_values_percent = [*radar_values_percent, radar_values_percent[0]]
        global_radar_chart.clear()
        radar_loc = np.linspace(start=0, stop=2 * np.pi, num=len(radar_values_percent))
        global_radar_chart.plot(radar_loc, radar_values_percent)
        global_radar_lines, global_radar_labels = global_radar_chart.set_thetagrids(np.degrees(radar_loc),
                                                                                    labels=radar_names)
        global_radar_chart.set_title(global_radar_chart_name)

    def last_5_min_radar_chart_upd():
        last_5_min_ago_map = emotions_map_time_list[0]
        if len(emotions_map_time_list) >= 300:
            last_5_min_ago_map = emotions_map_time_list[-300]
        sum_time_last_5_min = {}
        for k,v in emotions_map_time_list[-1].items():
            sum_time_last_5_min[k] = v - last_5_min_ago_map[k]
        radar_values = list(sum_time_last_5_min.values())
        radar_values_percent = list(map(lambda x: x / sum(radar_values), radar_values))
        radar_values_percent = [*radar_values_percent, radar_values_percent[0]]
        last_5_min_radar_chart.clear()
        radar_loc = np.linspace(start=0, stop=2 * np.pi, num=len(radar_values_percent))
        last_5_min_radar_chart.plot(radar_loc, radar_values_percent)
        last_5_min_radar_lines, last_5_min_radar_labels = last_5_min_radar_chart.set_thetagrids(np.degrees(radar_loc),
                                                                                                labels=radar_names)
        last_5_min_radar_chart.set_title(last_5_min_radar_chart_name)

    def update_plots():
        fig.canvas.draw()
        fig.canvas.flush_events()
        fig.canvas.draw_idle()

    plt.ion()

    # create plots

    fig, axs = plt.subplots(2)
    axBar = axs[0]
    axPie = axs[1]

    # bar chart
    x = 0.5 + np.arange(len(names))
    y = np.random.uniform(0, 1, len(names))
    width = 1
    barPlot = axBar.bar(x, y, width=width, edgecolor="white", linewidth=0.7)
    axBar.set_xticks(np.add(x, 0))
    axBar.set_xticklabels(names.values())

    # global pie chart
    pieValues = list(sum_time_global.values())
    pieKeys = list(sum_time_global.keys())
    axPie.pie(np.array(pieValues), labels=pieKeys)

    # lines chart create
    fig2, axs2 = plt.subplots(len(names), sharex=True)
    emotion_chart_list = []
    fig2.tight_layout()
    for i in range(len(names)):
        new_emotion_chart, = axs2[i].plot([0] * lines_chart_history_size, lines_data_x)
        axs2[i].set_ylim(0, 1)
        axs2[i].set_yticklabels([0.0])
        axs2[i].set_xlim(-1, 99)
        axs2[i].title.set_text(names[i])
        emotion_chart_list.append(new_emotion_chart)

    # radar chart create

    fig3, ax_radars = plt.subplots(2, subplot_kw={'projection': 'polar'})
    fig3.tight_layout()
    radar_names = list(names.values())
    radar_names = [*radar_names, radar_names[0]]
    radar_values = list(sum_time_global.values())
    radar_values = [*radar_values, radar_values[0]]
    radar_loc = np.linspace(start=0, stop=2 * np.pi, num=len(radar_values))
    global_radar_chart = ax_radars[0]
    global_radar_chart.plot(radar_loc, radar_values, label=global_radar_chart_name)
    global_radar_lines, global_radar_labels = global_radar_chart.set_thetagrids(np.degrees(radar_loc),
                                                                                labels=radar_names)
    global_radar_chart.set_title(global_radar_chart_name)

    last_5_min_radar_chart = ax_radars[1]
    last_5_min_radar_chart.plot(radar_loc, radar_values, label=global_radar_chart_name)
    last_5_min_radar_lines, last_5_min_radar_labels = global_radar_chart.set_thetagrids(np.degrees(radar_loc),
                                                                                        labels=radar_names)
    last_5_min_radar_chart.set_title(last_5_min_radar_chart_name)

    def upd_data_maps(prob):
        dta_list = prob.tolist()
        max_index = dta_list.index(max(dta_list))
        max_name = names[max_index]
        sum_time_global[max_name] += 1
        last_time_element = emotions_map_time_list[-1].copy()
        last_time_element[max_name] += 1
        emotions_map_time_list.append(last_time_element)

    def show_result(result_list, prob):
        upd_data_maps(prob)
        bar_chart_upd(prob)
        lines_chart_upd(prob)
        pie_chart_upd()
        radar_chart_upd()
        update_plots()

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, transforms=classify_transforms(imgsz[0]), vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, transforms=classify_transforms(imgsz[0]), vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.Tensor(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            results = model(im)

        # Post-process
        with dt[2]:
            pred = F.softmax(results, dim=1)  # probabilities

        # Process predictions
        for i, prob in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem)
            # + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt

            s += '%gx%g ' % im.shape[2:]  # print string
            annotator = Annotator(im0, example=str(names), pil=True)

            # Print results
            top5i = prob.argsort(0, descending=True)[:5].tolist()  # top 5 indices
            show_result(prob.argsort(0, descending=True).tolist(), prob)
            s += f"{', '.join(f'{names[j]} {prob[j]:.2f}' for j in top5i)}, "

            text_prestring = '' if dataset.mode == 'image' else (str(frame) + ";")

            # Write results
            text = text_prestring + ';'.join(f'{prob[j]:.2f} {names[j]}' for j in top5i)
            if save_img or view_img:  # Add bbox to image
                annotator.text((32, 32), text, txt_color=(255, 255, 255))
            if save_txt:  # Write to file
                with open(f'{txt_path}.txt', 'a') as f:
                    f.write(text + '\n')

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-cls.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[224], help='inference size h,w')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/predict-cls', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


def custom_main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
