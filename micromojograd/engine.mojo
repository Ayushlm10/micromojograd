struct Value(Writable):
    var data: Float64

    def __init__(out self, data: Float64):
        self.data = data

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Value(data=", self.data, ")")

    def __add__(self, other: Self) -> Self:
        return Self(self.data + other.data)

    def __mul__(self, other: Self) -> Self:
        return Self(self.data * other.data)
