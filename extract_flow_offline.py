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

# FRAME_EXTs = [
#     "jepg",
#     "jpg",
#     "png"
# ]


def compute_TVL1(prev, curr, TVL1, bound=20):
    """
    Args:
        prev (numpy.ndarray): a previous video frame, dimension is
            `height` x `width`.
        curr (numpy.ndarray):  a current video frame, dimension is
            `height` x `width`.
        bound (int): specify the maximum and minimux of optical flow.

    Return:
        flow (numpy.ndarray): optical flow.
    """
    # TVL1=cv2.optflow.DualTVL1OpticalFlow_create()
    # TVL1 = cv2.DualTVL1OpticalFlow_create()
    # TVL1=cv2.createOptFlow_DualTVL1()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0
    return flow


def compute_FB(prev, curr, bound=20):
    """
    Args:
        prev (numpy.ndarray): a previous video frame, dimension is
            `height` x `width`.
        curr (numpy.ndarray):  a current video frame, dimension is
            `height` x `width`.
        bound (int): specify the maximum and minimux of optical flow.

    Return:
        flow (numpy.ndarray): optical flow.
    """
    # flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.702, 5, 10, 2, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 5, 13, 10, 5,
                                        1.1, 0)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0
    return flow


def parse_frames_path(frame_path):
    '''
    Args:
        frame_path(str):  path to frames.
    Return:
        items(list): [sigle_video_path_1, sigle_video_path_2, ...]
    '''

    items = []
    dirlist = os.listdir(frame_path)
    for d in dirlist:
        d = os.path.join(frame_path, d)
        if os.path.isdir(d):
            items.append(d)

    return items


def save_flow(video_flows,
              flow_path,
              format="flow{}_{:05d}.{}",
              ext="jpg",
              separate=True):
    """
    Args:
        video_flows (list): store the flow (numpy.ndarray)
        flow_type (str): the path to store the flows.
        format (str): using which formate to store the flow images.
    Return:
    """
    if not os.path.exists(flow_path):
        os.makedirs(flow_path)
    for i, flow in enumerate(video_flows):
        if separate:
            cv2.imwrite(os.path.join(flow_path, format.format("_x", i, ext)),
                        flow[:, :, 0])

            cv2.imwrite(os.path.join(flow_path, format.format("_y", i, ext)),
                        flow[:, :, 1])
        else:
            # np.save(os.path.join(flow_path, format.format("", i, "npy")), flow)
            new_flows = np.zeros((flow.shape[0], flow.shape[1], 3),
                                 dtype=flow.dtype)
            new_flows[:, :, :2] = flow
            cv2.imwrite(os.path.join(flow_path, format.format("_xy", i, ext)),
                        new_flows)


def extract_flow(frames_path, flow_type="TVL1"):
    """
    Args:
        frames_path (str): frames path
        flow_type (str): TVL1 or FB.
    Returns:
        flows (list): store the flow (numpy.ndarray)
    """
    frames = glob.glob(os.path.join(frames_path, '*.*'))

    assert len(frames) != 0, "the number of frames is 0!!!!"

    frames.sort()
    flows = []
    # prev = Image.open(frames[0]).convert('L')
    # prev = np.array(prev)
    prev = cv2.imread(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    if "TVL1" in flow_type:
        TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()

    for i, frame_curr in enumerate(frames):
        if i == 0:
            continue
        # curr = Image.open(frame_curr).convert('L')
        # curr = np.array(curr)
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        if "TVL1" in flow_type:
            tmp_flow = compute_TVL1(prev, curr, TVL1)
        elif "FB" in flow_type:
            tmp_flow = compute_FB(prev, curr)
        else:
            raise NotImplementedError(
                "The flow type {} now is not supported.".format(flow_type))

        flows.append(tmp_flow)
        prev = curr

    # save_flow(flows, args.des_path, separate=True, ext=args.ext)
    if len(flows) == 0:
        return flows, [frames_path, "Extracting Failed"]
    # print(status)
    return flows, [frames_path, "Success"]


def extract_flow_wrapper(args, frames_path):
    flow, status = extract_flow(frames_path, args.flow_type)
    if "Success" in status[1]:
        flow_path = os.path.join(args.out_path, frames_path.split("/")[-1])
        save_flow(flow, flow_path, ext=args.out_ext)
        print(status[0])
    return status


def extract_flow_parallel(args):
    '''
    Args:
        args(argparse.Namespace):  contains some neccessary parameters.
    Return:
        items(list): [sigle_video_path_1, sigle_video_path_2, ...]
    '''

    start = time.time()
    items = parse_frames_path(args.root_path)
    print("{} items".format(len(items)))
    # exit()
    status_list = Parallel(n_jobs=args.num_jobs, backend="multiprocessing")(
        delayed(extract_flow_wrapper)(args, path) for path in items)
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
        "root_path",
        help="path to store the frames",
        type=str,
    )

    parser.add_argument(
        "-o",
        "--out_path",
        help="path to store processed videos",
        default="./tmpflow",
        type=str,
    )

    parser.add_argument(
        "--flow_type",
        help="using which methods to extract the flow (TVL1 or FB)",
        default="TVL1",
        choices=["TVL1", "FB"],
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
    parser.add_argument(
        "--width",
        help="width of videos",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--height",
        help="height of videos",
        default=0,
        type=int,
    )

    # parser.add_argument(
    #     "-e",
    #     "--frame_exts",
    #     help="extensions of video",
    #     nargs='+',
    #     default=None,
    #     type=str,
    # )

    parser.add_argument(
        "--out_ext",
        help="the extension of output files",
        type=str,
        default="jpg",
    )

    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    extract_flow_parallel(args)

    print("Done!")