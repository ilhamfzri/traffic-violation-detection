import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
import numpy as np
import os


def show_values(axs, orient="v", space=0.01):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height() * 0.01)
                value = "{}".format(int(p.get_height()))
                ax.text(_x, _y, value, ha="center")
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height() * 0.5)
                value = "{}".format(int(p.get_width()))
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)


def plot_vehicle_distribution(db_data, file_output):
    plt.figure().clear()
    data_distribution = {
        "Car": 0,
        "Motorcycle": 0,
        "Bus": 0,
        "Truck": 0,
        "Bicycle": 0,
    }
    for key in db_data.keys():
        vehicle_type = db_data[key]["vehicle_type"]
        data_distribution[vehicle_type] += 1

    sns.set_theme(style="darkgrid")
    y_data = list(data_distribution.values())
    x_data = list(data_distribution.keys())
    ax = sns.barplot(x=x_data, y=y_data)
    show_values(ax)

    plt.title("Violation Data\nVehicle Distribution", weight="bold")
    plt.savefig(file_output, dpi=300)


def plot_violation_distribution(db_data, file_output):
    plt.figure().clear()
    data_distribution = {
        "Wrong Way": 0,
        "Running Red Light": 0,
        "Helmet Violation": 0,
    }
    for key in db_data.keys():
        violation_type = db_data[key]["violation_type"]
        data_distribution[violation_type] += 1

    sns.set_theme(style="darkgrid")
    y_data = list(data_distribution.values())
    x_data = list(data_distribution.keys())
    ax2 = sns.barplot(x=x_data, y=y_data)
    show_values(ax2)

    plt.title("Violation Data\nViolation Type Distribution", weight="bold")
    plt.savefig(file_output, dpi=300)


def get_parser():
    parser = argparse.ArgumentParser(description="Create Data Statistics For Analysis")
    parser.add_argument(
        "--json_path", type=str, default=None, help="Database JSON Path"
    )

    return parser


def main():
    print("Here")
    argparse = get_parser().parse_args()
    file_path = argparse.json_path

    with open(file_path, "r", encoding="utf-8") as db_file:
        db_json_data = json.load(db_file)
        db_data = {}

        for violation_type in db_json_data.keys():
            violation_data = db_json_data[violation_type]
            for object_id in violation_data.keys():
                print(violation_type)
                id_data = violation_data[object_id]["img_proof"][-14:-4]

                if violation_type == "running_red_light":
                    print("here")
                    id_data = id_data + "R"
                    violation_type_new = "Running Red Light"

                elif violation_type == "wrong_way ":
                    id_data = id_data + "W"
                    violation_type_new = "Wrong Way"

                db_data[id_data] = {
                    "vehicle_type": violation_data[object_id]["vehicle_type"],
                    "violation_type": violation_type_new,
                    "img_proof": violation_data[object_id]["img_proof"],
                    "timestamp": violation_data[object_id]["timestamp"],
                }
        vehicle_distribution_path = os.path.join(
            os.path.dirname(file_path), "vehicle_distribution_plot.png"
        )
        plot_vehicle_distribution(db_data, vehicle_distribution_path)

        violation_distribution_path = os.path.join(
            os.path.dirname(file_path), "violation_distribution_plot.png"
        )
        plot_violation_distribution(db_data, violation_distribution_path)


if __name__ == "__main__":
    main()
