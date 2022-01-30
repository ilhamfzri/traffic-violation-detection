import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

# KITTI Instances per Category
kitti = {"mobil": 31656, "motor": 0, "bus": 511, "truk": 1094, "sepeda": 1627}

# KITTI Image per Category
kitti_image_per_category = {
    "1": 68.68337654854508,
    "2": 28.377989052146358,
    "3": 2.938634399308557,
    "4": 0.0,
    "5": 0.0,
}

# KITTI Image per Instance
kitti_image_per_instance = {
    1: 12.028233938346299,
    2: 14.909248055315471,
    3: 13.281475079227889,
    4: 12.662057044079516,
    5: 11.07749927974647,
    6: 8.369346009795448,
    7: 7.101699798329012,
    8: 4.868913857677903,
    9: 4.638432728320368,
    10: 3.2987611639297034,
    11: 1.930279458369346,
    12: 1.3684817055603573,
    13: 1.7430135407663496,
    14: 1.152405646787669,
    15: 0.9219245174301354,
    16: 0.41774704696053006,
    17: 0.08643042350907519,
    18: 0.02881014116969173,
    19: 0.043215211754537596,
    20: 0.05762028233938346,
    21: 0.014405070584845865,
}


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


def dataset_distribution_bar(data):
    sns.set_theme(style="darkgrid")
    tips = sns.load_dataset("tips")
    y_data = list(data.values())
    x_data = list(data.keys())

    ax = sns.barplot(x=x_data, y=y_data, data=tips)

    show_values(ax)

    # ax.text("motor".name, "motor".tip, 5, color="black", ha="center")
    plt.title("KITTI Dataset\nDistribusi Jumlah Objek Per Kategori", weight="bold")
    plt.show()


def dataset_distribution_line(data):
    sns.set_style("darkgrid")
    y_data = list(data.values())
    x_data = list(data.keys())

    ax = sns.lineplot(x=x_data, y=y_data, palette="hot", marker="o")
    ax.set(xlabel="Jumlah Kategori", ylabel="Persentase Gambar")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.title("KITTI Dataset\nGambar Per Kategori", weight="bold")
    plt.show()


def dataset_distribution_line2(data):
    sns.set_style("darkgrid")
    y_data = list(data.values())
    x_data = list(data.keys())

    ax = sns.lineplot(x=x_data, y=y_data, palette="hot", marker="o")
    ax.set(xlabel="Jumlah Objek", ylabel="Persentase Gambar")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.title("KITTI Dataset\nGambar Per Objek", weight="bold")
    plt.show()


if __name__ == "__main__":
    # KITTI Instances per Category Plot
    # dataset_distribution_bar(kitti)

    # KITTI Image per Category
    # dataset_distribution_line(kitti_image_per_category)

    # KITTI Instance per Category
    dataset_distribution_line2(kitti_image_per_instance)
