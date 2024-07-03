import gdb
import gdb.printing

class UniquePtrPrinter:
    "Print a std::unique_ptr"

    def __init__(self, val):
        self.val = val

    def to_string(self):
        element_type = self.val.type.template_argument(0)
        pointer = self.val['_M_t']['_M_head_impl']
        if pointer == 0:
            return 'std::unique_ptr<{}>(nullptr)'.format(element_type)
        else:
            return 'std::unique_ptr<{}>({})'.format(element_type, pointer)