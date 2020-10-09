import os
import argparse
import subprocess
import glob
from joblib import Parallel
from joblib import delayed
import time
import datetime

def process_videos(args, src_path):
    if args.same_dir:
        vid_name = src_path[len(args.root_path):]
    else:
        vid_name = src_path.split("/")[-1]
    dest_path = os.path.join(args.des_path, vid_name)
    if os.path.exists(dest_path):
        print("{}: {}".format(vid_name, "Exists"))
        return vid_name, "Exists"

    if not os.path.exists(os.path.dirname(dest_path)):
        os.makedirs(os.path.dirname(dest_path))

    command = ["ffmpeg",
    "-i", "'%s'" % src_path,
    "-vf scale=%d:%d" % (args.width, args.height),
    "-q 1", 
    "-crf 18",
    "-an" if not args.voice else "",
    "'%s'" % dest_path]

    command = ' '.join(command)
    status = False
    try:
        _ = subprocess.check_output(
                    command, 
                    shell=True,
                    stderr=subprocess.STDOUT
                )

    except subprocess.CalledProcessError as err:
        print("{}: {}".format(vid_name, err.output))
        return vid_name, err.output

    if not os.path.exists(dest_path):
        print("{}: {}".format(vid_name, "Failed"))
        return vid_name, "Failed"

    print("{}: {}".format(vid_name, "Success"))
    return vid_name, "Success"

def process_videos_wrapper(args):
    start = time.time()
    items = glob.glob(os.path.join(args.root_path, "*.mp4"))
    if len(items) == 0:
        items = glob.glob(os.path.join(args.root_path, "*", "*.mp4"))

    status_list = Parallel(n_jobs=args.num_jobs)(
                    delayed(process_videos)(
                        args, src_path
                    ) for src_path in items
                    
    )
    # print(status_list)
    cnt = 0
    with open("log.csv", "w") as f:
        for item in status_list:
            line = " ".join(str(a) for a in item) 
            f.write(line)
            f.write("\n")
            if item[1] not in["Success", "Exists" ]:
                cnt += 1

        total_time = datetime.timedelta(seconds=int(time.time()-start))
        end_lines = [
            "{} videos failed to process.".format(cnt),
            "Spend {} to process data".format(total_time)
        ]
        end_lines = "\n".join(end_lines)
        print(end_lines)
        f.write(end_lines)

    # process_videos('', args)
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(
        description="Resizing the videos to the specific resolution."
    )
    parser.add_argument(
        "-r",
        "--root_path",
        help="path to store raw videos",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-d",
        "--des_path",
        help="path to store processed videos",
        default="./outputs",
        type=str,
    )
    """
        the content of file_list.csv like:
        vid1_path 
        vid2_path
        vid3_path
        ...
    """

    parser.add_argument(
        "-j", 
        "--num_jobs",
        help="the number of job",
        default=2,
        type=int,
    )

    # default settings of resizing video are w = 320 and height = -1 
    # which is supposed to keep the aspect ratio consistent with raw video.
    # I think that choice is the most appropriate.
    parser.add_argument(
        "--width",
        help="width of videos",
        default=320,
        type=int,
    )

    parser.add_argument(
        "--height",
        help="height of videos",
        default=-2,
        type=int,
    )

    parser.add_argument(
        "-v", 
        "--voice",
        help="whether or not to remove the voice in video",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--same_dir",
        help="keep the dir tree consistent with raw videos",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    if not os.path.exists(args.des_path):
        os.makedirs(args.des_path)

    process_videos_wrapper(args)
