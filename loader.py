import cv2
import numpy as np
from os import listdir


class ObjectLabel:
    def __init__(self, object_class,  x_min, y_min, x_max, y_max, distance):
        self.object_class = object_class
        assert x_min <= x_max
        assert y_min <= y_max
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.distance = distance

    @classmethod
    def parse_from_line(cls, line_str):
        components = line_str.strip().split()
        object_class = components[0]
        x_min = float(components[1])
        y_min = float(components[2])
        x_max = float(components[3])
        y_max = float(components[4])
        distance = float(components[5])
        return cls(object_class, x_min, y_min, x_max, y_max, distance)


class LabeledImage:
    def __init__(self, name, intrinsic_matrix, image_data, object_labels: list[ObjectLabel]):
        self.name = name
        self.intrinsic_matrix = intrinsic_matrix
        self.image_data = image_data
        self.object_labels = object_labels

    @staticmethod
    def file_lines(file) -> list[str]:
        return [line.strip() for line in file.readlines()]

    @classmethod
    def create_from_file_name(cls, name, matrix_dir, image_dir, label_dir):
        matrix_file = open(matrix_dir + '/' + name + '.txt')
        image_file = cv2.imread(image_dir + '/' + name + '.png', cv2.IMREAD_COLOR)
        label_file = open(label_dir + '/' + name + '.txt')

        intrinsic_matrix = np.array([list(map(float, line.split())) for line in cls.file_lines(matrix_file)])
        object_labels = [ObjectLabel.parse_from_line(line_str) for line_str in cls.file_lines(label_file)]

        return cls(name, intrinsic_matrix, image_file, object_labels)


class DataSet:
    def __init__(self, labeled_images: list[LabeledImage]):
        self.labeled_images = labeled_images

    @classmethod
    def load_from_directory(cls, directory: str):
        matrix_dir = directory + '/calib'
        image_dir = directory + '/images'
        label_dir = directory + '/labels'
        image_files = listdir(image_dir)
        labeled_images = []
        for file in image_files:
            name = file[0:len(file) - 4]
            labeled_images.append(LabeledImage.create_from_file_name(name, matrix_dir, image_dir, label_dir))
        return cls(labeled_images)
