import argparse
import cv2
import sys

sys.path.append("yolov5_repo")

from tqdm import tqdm
from tvdr.utils.params import Parameter
from tvdr.utils.config import ConfigLoader
from tvdr.utils.path import create_folder
from tvdr.core.pipelines import TrafficViolationDetectionPipelines


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
        "--output_dir", type=str, default="result", help="inference_result"
    )
    return parser


def main():
    args = parser().parse_args()
    create_folder(args.output_dir)

    # Load configuration
    cfg = ConfigLoader()
    parameter = Parameter()
    parameter = cfg.load_parser(args.config_file)
    parameter.select_device = args.device

    # Load video path
    if args.video_path is not None:
        video_path = args.video_path
    else:
        if parameter.video_path is not None:
            video_path = parameter.video_path
        else:
            print("ERROR : Please set video path correctly!")
            return 0

    # Set Init Pipelines
    pipeline = TrafficViolationDetectionPipelines(parameter)
    pipeline.update_parameter(parameter)
    pipeline.vr.video_recorder_init(video_path, args.output_dir)

    # Read video frames
    vid = cv2.VideoCapture(video_path)
    if vid.isOpened() == False:
        print("Error opening video file!")

    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(0, 100)):
        vid.set(1, i)
        _, frame = vid.read()
        result_frame = pipeline.update(frame)
        pipeline.vr.video_recorder_update(result_frame)


if __name__ == "__main__":
    main()
