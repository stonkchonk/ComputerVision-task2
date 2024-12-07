class CarLabel:
    def __init__(self, object_class,  x_min, x_max, y_min, y_max, distance):
        self.object_class = object_class
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
        x_max = float(components[2])
        y_min = float(components[3])
        y_max = float(components[4])
        distance = float(components[5])
        return cls(object_class, x_min, x_max, y_min, y_max, distance)


def read_file_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Strip newline characters from each line
    return [line.strip() for line in lines]


