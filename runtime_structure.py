import struct

class structure(object):
    def __init__(self, path):
        self.module = []
        self.function = []
        self.table = []
        self.memory = []
        self.globall = []
        self.export = []
        