import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

# COCO Instances per Category
coco = {"mobil": 45799, "motor": 9096, "bus": 6354, "truk": 10388, "sepeda": 7429}

# COCO Image per Category
coco_image_per_category = {
    "1": 62.74661883755878,
    "2": 28.70715982354937,
    "3": 7.261621988462844,
    "4": 1.168258277182607,
    "5": 0.1163410732464007,
}

# COCO Image per Instance
coco_image_per_instance = {
    1: 33.7922342333608,
    2: 17.460856076397306,
    3: 11.87648456057007,
    4: 8.35716709486645,
    5: 6.500557467642638,
    6: 4.561539580202627,
    7: 3.38843375830142,
    8: 2.7243201318532164,
    9: 2.4431625381744144,
    10: 1.7790489117262105,
    11: 1.4397207814242086,
    12: 1.1828009113384073,
    13: 0.998594212031606,
    14: 1.2118861796500073,
    15: 0.7610645208202046,
    16: 0.4459741141112027,
    17: 0.25207232536720153,
    18: 0.22783460177420137,
    19: 0.15512143099520093,
    20: 0.09695089437200058,
    21: 0.05817053662320035,
    22: 0.0678656260604004,
    23: 0.05817053662320035,
    24: 0.0339328130302002,
    25: 0.009695089437200058,
    26: 0.019390178874400116,
    27: 0.014542634155800087,
    28: 0.024237723593000145,
    29: 0.004847544718600029,
    30: 0.029085268311600174,
    31: 0.009695089437200058,
    33: 0.004847544718600029,
    38: 0.004847544718600029,
    40: 0.004847544718600029,
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
    plt.title("MS COCO Dataset\nDistribusi Jumlah Objek Per Kategori", weight="bold")
    plt.show()


def dataset_distribution_line(data):
    sns.set_style("darkgrid")
    y_data = list(data.values())
    x_data = list(data.keys())

    ax = sns.lineplot(x=x_data, y=y_data, palette="hot", marker="o")
    ax.set(xlabel="Jumlah Kategori", ylabel="Persentase Gambar")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.title("MS COCO Dataset\nGambar Per Kategori", weight="bold")
    plt.show()


def dataset_distribution_line2(data):
    sns.set_style("darkgrid")
    y_data = list(data.values())
    x_data = list(data.keys())

    ax = sns.lineplot(x=x_data, y=y_data, palette="hot", marker="o")
    ax.set(xlabel="Jumlah Objek", ylabel="Persentase Gambar")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.title("MS COCO Dataset\nGambar Per Objek", weight="bold")
    plt.show()


if __name__ == "__main__":
    # COCO Instances per Category Plot
    # dataset_distribution_bar(coco)

    # COCO Image per Category
    # dataset_distribution_line(coco_image_per_category)

    # COCO Instance per Category
    dataset_distribution_line2(coco_image_per_instance)
