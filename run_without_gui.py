import argparse
import cv2
import sys
import time
import logging

sys.path.append("yolov5_repo")

from tqdm import tqdm
from tvdr.utils.params import Parameter
from tvdr.utils.config import ConfigLoader
from tvdr.utils.path import create_folder
from tvdr.core.pipelines import TrafficViolationDetectionPipelines

logging.basicConfig()

logger = "tvdr"
logger = logging.getLogger("tvdr")
logger.setLevel(logging.INFO)


def parser():
    parser = argparse.ArgumentParser(
        description="Traffic Violation Detection Without GUI"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="set this path to configuration file",
    )
    parser.add_argument("--video_path", type=str, default=None, help="set video path")
    parser.add_argument(
        "--device", type=str, default="cpu", help="set this to gpu if you have cuda"
    )
    parser.add_argument(
        "--output_path", type=str, default="result", help="inference_result"
    )
    return parser


def main():
    args = parser().parse_args()
    create_folder(args.output_dir)

    checkpoint_time = time.time()
    # Load configuration
    cfg = ConfigLoader()
    parameter = Parameter()
    parameter = cfg.load_parser(args.config_file)
    parameter.device = args.device

    # Load video path
    if args.video_path is not None:
        video_path = args.video_path
    else:
        if parameter.device is not None:
            video_path = parameter.video_path
        else:
            print("ERROR : Please set video path correctly!")
            return 0

    parameter.video_path = args.video_path

    # Set Init Pipelines
    pipeline = TrafficViolationDetectionPipelines(parameter)
    pipeline.update_parameter(parameter)
    pipeline.vr.video_recorder_init(video_path, args.output_dir)

    # Read video frames
    vid = cv2.VideoCapture(video_path)
    if vid.isOpened() == False:
        print("Error opening video file!")

    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(0, frame_count)):
        _, frame = vid.read()
        result_frame = pipeline.update(frame, i)
        checkpoint = time.time()
        pipeline.vr.video_recorder_update(result_frame)
        logger.debug(
            f"Video Recorder Process Time : {(time.time()-checkpoint)*1000:.1f}ms"
        )

    print(f"Total Processing Time : {time.time()-checkpoint_time}s")


if __name__ == "__main__":
    main()
