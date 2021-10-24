from tvdr.interface.main import MainWindows
from tvdr.utils import Parameter


def main():
    x = Parameter()
    for index in x.yolo_model_dict.keys():
        print(index)
        print(x.yolo_model_dict[index])


if __name__ == "__main__":
    main()
