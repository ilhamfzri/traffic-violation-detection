import json
from tvdr.core import (
    VehicleDetectionConfig,
    RunningRedLightConfig,
    HelmetViolationConfig,
)


class PipelineConfig:
    def __init__(
        self,
        video_path: str = None,
        detect_rrl: bool = True,
        detect_hvd: bool = True,
        detect_wr: bool = True,
        write_db: bool = True,
    ):
        self.video_path = video_path
        self.detect_rrl = detect_rrl
        self.detect_hvd = detect_hvd
        self.detect_wr = detect_wr

        self.vd_config = VehicleDetectionConfig()
        self.rrl_config = RunningRedLightConfig()
        self.hv_config = HelmetViolationConfig()

    def save_config(self, path: str):
        config_dict = {}
        config_atr = self.__dict__.keys()
        for atr in config_atr:
            if "config" not in atr:
                config_dict[atr] = getattr(self, atr)
            else:
                sub = getattr(self, atr)
                sub_config_atr = sub.__dict__.keys()
                sub_config_dict = {}

                for sub_atr in sub_config_atr:
                    sub_config_dict[sub_atr] = getattr(sub, sub_atr)

                if atr == "vd_config":
                    config_dict["vehicle_detection"] = sub_config_dict

                elif atr == "rrl_config":
                    config_dict["running_red_light"] = sub_config_dict

                elif atr == "hv_config":
                    config_dict["helmet_violation"] = sub_config_dict

        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)

    def load_config(self, path: str):
        sub_attr = ["vehicle_detection", "running_red_light", "helmet_violation"]
        with open(path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        for atr in config_dict.keys():
            if atr in sub_attr:
                sub_config_dict = config_dict[atr]
                for sub_atr in sub_config_dict.keys():
                    if atr == "vehicle_detection":
                        setattr(self.vd_config, sub_atr, sub_config_dict[sub_atr])

                    elif atr == "running_red_light":
                        setattr(self.rrl_config, sub_atr, sub_config_dict[sub_atr])

                    elif atr == "helmet_violation":
                        setattr(self.hv_config, sub_atr, sub_config_dict[sub_atr])

            else:
                setattr(self, atr, config_dict[atr])
