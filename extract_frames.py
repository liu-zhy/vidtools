# coding: utf-8
import numpy as np
import cv2
import os
import time
import argparse
import subprocess
import glob
from joblib import Parallel
from joblib import delayed
import time
import datetime

VIDEO_EXTs = [
    "mp4",
    "avi",
    "mkv",
    "webm",
]


def save_imgs(video_imgs, img_path, format="img_{:05d}.{}", ext="jpg"):
    """
    Args:
        video_imgs (list): contains the frames.
        img_path (str):  the path to store the extracted frames.
        width (int): set new width for frames.
        height (int): set new height for frames.
        format (str): using which format to store the flow images.
        ext (str): the extension of file. default: `jpg`.
    Return:
        flag (bool): denote the status when saving images. True is success and False is failure.
    """
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    # img_path += "/" + video_name\
    for i, img in enumerate(video_imgs, 1):
        if not cv2.imwrite(os.path.join(img_path, format.format(i, ext)), img):
            return False
    return True


def resize_frames(frames, new_size_scheme):
    """
    Args:
        frames (list): the elements in frames are numpy.ndarray.
        new_size_scheme (str):  resize scheme.
    Return:
        frames: the elements in list are resized frames.
    """
    h, w, _ = frames[0].shape
    # new_w, new_h = w, h
    if new_size_scheme.startswith("min"):
        min_edge = int(new_size_scheme.split("=")[1])
        scale_ratio = min_edge / min(w, h)
        new_h = int(scale_ratio * h)
        new_w = int(scale_ratio * w)
    elif new_size_scheme.find(":") != -1:
        new_w = int(new_size_scheme.split(":")[0])
        new_h = int(new_size_scheme.split(":")[1])

    if (w, h) == (new_w, new_h):
        return frames

    new_frames = []
    for img in frames:
        new_frames.append(
            cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR))

    return new_frames


def extract_frames(video_path, interval=1, size=None):
    """
    Args:
        video_path (str): denote the video path.
        interval (int):  sampling interval.
        size (str): the scheme to resize the images.
    Return:
        frames (list): the elements in list are numpy.ndarray.
        status (list): [video_path, FLAG]
    """
    # start = time.time()
    cap = cv2.VideoCapture(video_path)
    imgs = []
    # return true if successful
    if cap.isOpened() != True:
        status = [video_path, "Openning Failed"]
        return imgs, status
    index = 0
    while True:
        if interval > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            index += interval

        ret, frame = cap.read()
        if ret != True:
            break
        imgs.append(frame)

    cap.release()
    if len(imgs) == 0:
        return imgs, [video_path, "Decoding Failed"]
    if size is not None and size.strip() != "":
        imgs = resize_frames(imgs, size)
    return imgs, [video_path, "Success"]


def parse_video_path(args):
    '''
    Args:
        args(argparse.Namespace):  contains some neccessary parameters.
    Return:
        items(list): [sigle_video_path_1, sigle_video_path_2, ...]
    '''

    items = []
    for ext in args.videos_exts:
        items.extend(
            glob.glob(os.path.join(args.video_path, "*.{}".format(ext))))
    if len(items) == 0:
        for ext in args.videos_exts:
            items.extend(
                glob.glob(
                    os.path.join(args.video_path, "*", "*.{}".format(ext))))
    return items


def extract_frames_wrapper(args, video_path):
    '''
    Args:
        args(argparse.Namespace):  contains some neccessary parameters.
    Return:
        status(list): [video_path, flag]
    '''
    imgs, status = extract_frames(video_path, args.interval, args.size)
    if "Success" in status[1]:
        img_path = os.path.join(
            args.out_path,
            os.path.splitext(os.path.basename(video_path))[0])
        if save_imgs(imgs, img_path, ext=args.out_ext):
            print(status)
            return status
        else:
            print([status[0], "Saving Failed"])
            return [status[0], "Saving Failed"]
    else:
        print(status)
        return status


def extract_frames_parallel(args):
    '''
    Args:
        args(argparse.Namespace):  contains some neccessary parameters.
    Return:
        items(list): [sigle_video_path_1, sigle_video_path_2, ...]
    '''

    start = time.time()
    if args.videos_exts is None or len(args.videos_exts) == 0:
        args.videos_exts = VIDEO_EXTs
    else:
        args.videos_exts = set(args.videos_exts)
        args.videos_exts = list(args.videos_exts)

    items = parse_video_path(args)
    print("{} videos".format(len(items)))

    status_list = Parallel(n_jobs=args.num_jobs)(
        delayed(extract_frames_wrapper)(args, src_path) for src_path in items)
    # print(status_list)
    cnt = 0
    with open("log.csv", "w") as f:
        for item in status_list:
            line = " ".join(str(a) for a in item)
            f.write(line)
            f.write("\n")
            if item[1] not in ["Success"]:
                cnt += 1

        total_time = datetime.timedelta(seconds=int(time.time() - start))
        end_lines = [
            "{} video(s) failed to process.".format(cnt),
            "Spend {} to process data".format(total_time)
        ]
        end_lines = "\n".join(end_lines)
        print(end_lines)
        f.write(end_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracting the frames from given videos.")
    parser.add_argument(
        "video_path",
        help="path to store the videos",
        type=str,
    )

    parser.add_argument(
        "-o",
        "--out_path",
        help="path to store processed videos",
        default="./tmp",
        type=str,
    )

    parser.add_argument(
        "-i",
        "--interval",
        help="sampling interval when decoding videos",
        default=1,
        type=int,
    )

    parser.add_argument(
        "-j",
        "--num_jobs",
        help="the number of job",
        default=2,
        type=int,
    )

    # default settings of resizing video are w = 0 and height = 0
    # which is the same with raw video.
    # parser.add_argument(
    #     "--width",
    #     help="width of videos",
    #     default=0,
    #     type=int,
    # )

    # parser.add_argument(
    #     "--height",
    #     help="height of videos",
    #     default=0,
    #     type=int,
    # )

    # resize="width:height" or "min=256"
    parser.add_argument(
        "--size",
        help="resize frames",
        default=None,
        type=str,
    )

    parser.add_argument(
        "-e",
        "--videos_exts",
        help="extensions of video",
        nargs='+',
        default=None,
        type=str,
    )

    parser.add_argument(
        "--out_ext",
        type=str,
        default="jpg",
    )

    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    extract_frames_parallel(args)

    print("Done!")