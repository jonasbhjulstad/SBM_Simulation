import gdb
import gdb.printing

class UniquePtrPrinter:
    "Print a std::vector"

    def __init__(self, val):
        self.val = val

    def to_string(self):
        element_type = self.val.type.template_argument(0)
        pointer = self.val['_M_t']['_M_head_impl']
        if pointer == 0:
            return 'std::vector<{}>(nullptr)'.format(element_type)
        else:
            return 'std::vector<{}>({})'.format(element_type, pointer)