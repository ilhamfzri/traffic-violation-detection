import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

# GTA V Instances per Category
gtav = {"mobil": 813, "motor": 918, "bus": 26, "truk": 135, "sepeda": 6}

# GTA V Image per Category
gtav_image_per_category = {
    "1": 32.59423503325942,
    "2": 48.78048780487805,
    "3": 16.186252771618626,
    "4": 2.4390243902439024,
    "5": 0.0,
}

# GTA V Image per Instance
gtav_image_per_instance = {
    1: 14.855875831485587,
    2: 18.403547671840354,
    3: 15.299334811529933,
    4: 13.303769401330376,
    5: 11.973392461197339,
    6: 8.647450110864744,
    7: 6.651884700665188,
    8: 2.882483370288248,
    9: 2.4390243902439024,
    10: 2.6607538802660753,
    11: 0.4434589800443459,
    12: 0.6651884700665188,
    13: 0.22172949002217296,
    14: 0.22172949002217296,
    15: 0.6651884700665188,
    16: 0.22172949002217296,
    17: 0.22172949002217296,
    18: 0.22172949002217296,
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
    plt.title(
        "GTA V Vehicle Dataset\nDistribusi Jumlah Objek Per Kategori", weight="bold"
    )
    plt.show()


def dataset_distribution_line(data):
    sns.set_style("darkgrid")
    y_data = list(data.values())
    x_data = list(data.keys())

    ax = sns.lineplot(x=x_data, y=y_data, palette="hot", marker="o")
    ax.set(xlabel="Jumlah Kategori", ylabel="Persentase Gambar")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.title("GTA V Vehicle Dataset\nGambar Per Kategori", weight="bold")
    plt.show()


def dataset_distribution_line2(data):
    sns.set_style("darkgrid")
    y_data = list(data.values())
    x_data = list(data.keys())

    ax = sns.lineplot(x=x_data, y=y_data, palette="hot", marker="o")
    ax.set(xlabel="Jumlah Objek", ylabel="Persentase Gambar")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.title("GTA V Vehicle Dataset\nGambar Per Objek", weight="bold")
    plt.show()


if __name__ == "__main__":
    # GTAV Instances per Category Plot
    dataset_distribution_bar(gtav)

    # GTAV Image per Category
    dataset_distribution_line(gtav_image_per_category)

    # GTAV Instance per Category
    dataset_distribution_line2(gtav_image_per_instance)
