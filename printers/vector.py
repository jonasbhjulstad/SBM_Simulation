import gdb.printing  # nolint


class LinVec1DPrinter:
    def __init__(self, vec):
        # val is the python representation of you C++ variable.
        # It is a "gdb.Value" object and you can query the member
        # atributes of the C++ object as below. Since the result is
        # another "gdb.Value" I'am converting it to a python float
        self.x = float(vec['x'])
        self.y = float(vec['y'])
        self.z = float(vec['z'])

    # Whatever the `to_string` method returns is what will be printed in
    # gdb when this pretty-printer is used
    def to_string(self):
        return "Coordinate(x={:.2G}, y={:.2G}, z={:.2G})".format(self.x, self.y, self.z)


# Create a "collection" of pretty-printers
# Note that the argument passed to "RegexpCollectionPrettyPrinter" is the name of the pretty-printer and you can choose your own
pp = gdb.printing.RegexpCollectionPrettyPrinter('cppsim')
# Register a pretty-printer for the Coordinate class. The second argument is a
# regular expression and my Coordinate class is in a namespace called `cppsim`
pp.add_printer('Coordinate', '^cppsim::Coordinate$', LinVec1DPrinter)
# Register our collection into GDB
gdb.printing.register_pretty_printer(gdb.current_objfile(), pp, replace=True)
