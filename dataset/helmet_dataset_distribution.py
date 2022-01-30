import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

# Helm Instances per Category
helm = {"Pakai Helm": 1198, "Tidak Pakai Helm": 777}

# Helm Image per Category
helm_image_per_category = {"1": 88.40125391849529, "2": 11.598746081504702}

# Helm Image per Instance
helm_image_per_instance = {
    1: 49.895833333333336,
    2: 25.104166666666668,
    3: 9.6875,
    4: 6.5625,
    5: 4.270833333333333,
    6: 2.083333333333333,
    7: 1.25,
    8: 0.20833333333333334,
    9: 0.3125,
    10: 0.20833333333333334,
    11: 0.10416666666666667,
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
    plt.title("Helmet Dataset\nDistribusi Jumlah Objek Per Kategori", weight="bold")
    plt.show()


def dataset_distribution_line(data):
    sns.set_style("darkgrid")
    y_data = list(data.values())
    x_data = list(data.keys())

    ax = sns.barplot(x=x_data, y=y_data)
    ax.set(xlabel="Jumlah Kategori", ylabel="Persentase Gambar")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.title("Helmet Dataset\nGambar Per Kategori", weight="bold")
    plt.show()


def dataset_distribution_line2(data):
    sns.set_style("darkgrid")
    y_data = list(data.values())
    x_data = list(data.keys())

    ax = sns.lineplot(x=x_data, y=y_data, palette="hot", marker="o")
    ax.set(xlabel="Jumlah Objek", ylabel="Persentase Gambar")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.title("Helmet Dataset\nGambar Per Objek", weight="bold")
    plt.show()


if __name__ == "__main__":
    # Helm Instances per Category Plot
    dataset_distribution_bar(helm)

    # Helm Image per Category
    dataset_distribution_line(helm_image_per_category)

    # Helm Instance per Category
    dataset_distribution_line2(helm_image_per_instance)
