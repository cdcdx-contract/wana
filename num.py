import io
import math
import struct
import ctypes
import z3
import utils

u8 = i8 = u16 = i16 = u32 = i32 = u64 = i64 = int
f32 = f64 = float

def int2u8(i: int) -> u8:
    return i & 0xFF

def int2i8(i: int) -> i8:
    i = i & 0xFF
    if i & 0x80:
        return i - 0x100
    return i

def int2u16(i: int) -> u16:
    return i & 0xFFFFFFFF

def int2i16(i: int) -> i16:
    i = i & 0xFFFF
    if i & 0x8000:
        return i - 0x10000
    return i

def int2u32(i: int) -> u32:
    return i & 0xFFFFFFFF

def int2i32(i: int) -> i32:
    i = i & 0xFFFFFFFF
    if i & 0x80000000:
        return i - 0x100000000
    return i

def int2u64(i: int) -> u64:
    return i & 0xFFFFFFFFFFFFFFFF

def int2i64(i: int) -> i64:
    i = i & 0xFFFFFFFFFFFFFFFF
    if i & 0x8000000000000000:
        return i - 0x10000000000000000
    return i

def i322f32(i: i32) -> f32:
    i = int2i32(i)
    return struct.unpack('<f', struct.pack('<i', i))[0]

def f322i32(f: f32) -> i32:
    return struct.unpack('<i', struct.pack('<f', f))[0]

def i642f64(i: i64) -> f64:
    i = int2i64(i)
    return struct.unpack('<i', struct.pack('<q', i))[0]

def f642i64(f: f64) -> i64:
    return struct.unpack('<q', struct.pack('<d', f))[0]


class LittleEndian:
    @staticmethod
    def u8(r: bytes):
        return struct.unpack('<B', r)[0]

    @staticmethod
    def i8(r: bytes):
        return struct.unpack('<b', r)[0]

    @staticmethod
    def u16(r: bytes):
        return struct.unpack('<H', r)[0]

    @staticmethod
    def i16(r: bytes):
        return struct.unpack('<h', r)[0]

    @staticmethod
    def u32(r: bytes):
        return struct.unpack('<I', r)[0]

    @staticmethod
    def i32(r: bytes):
        return struct.unpack('<i', r)[0]

    @staticmethod
    def u64(r: bytes):
        return struct.unpack('<Q', r)[0]

    @staticmethod
    def i64(r: bytes):
        return struct.unpack('<q', r)[0]

    @staticmethod
    def f32(r: bytes):
        return struct.unpack('<f', r)[0]

    @staticmethod
    def f64(r: bytes):
        return struct.unpack('<d', r)[0]

    @staticmethod
    def pack_u8(n: u8):
        return struct.pack('<B', n)

    @staticmethod
    def pack_i8(n: i8):
        return struct.pack('<b', n)

    @staticmethod
    def pack_u16(n: u16):
        return struct.pack('<H', n)

    @staticmethod
    def pack_i16(n: i16):
        return struct.pack('<h', n)

    @staticmethod
    def pack_u32(n: u32):
        return struct.pack('<I', n)

    @staticmethod
    def pack_i32(n: i32):
        return struct.pack('<i', n)

    @staticmethod
    def pack_u64(n: u64):
        return struct.pack('<Q', n)

    @staticmethod
    def pack_i64(n: i64):
        return struct.pack('<q', n)

    @staticmethod
    def pack_f32(n: f32):
        return struct.pack('<f', n)

    @staticmethod
    def pack_f64(n: f64):
        return struct.pack('<d', n)


class BigEndian:
    @staticmethod
    def u8(r: bytes):
        return struct.unpack('>B', r)[0]

    @staticmethod
    def i8(r: bytes):
        return struct.unpack('>b', r)[0]

    @staticmethod
    def u16(r: bytes):
        return struct.unpack('>H', r)[0]

    @staticmethod
    def i16(r: bytes):
        return struct.unpack('>h', r)[0]

    @staticmethod
    def u32(r: bytes):
        return struct.unpack('>I', r)[0]

    @staticmethod
    def i32(r: bytes):
        return struct.unpack('>i', r)[0]

    @staticmethod
    def u64(r: bytes):
        return struct.unpack('>Q', r)[0]

    @staticmethod
    def i64(r: bytes):
        return struct.unpack('>q', r)[0]

    @staticmethod
    def f32(r: bytes):
        return struct.unpack('>f', r)[0]

    @staticmethod
    def f64(r: bytes):
        return struct.unpack('>d', r)[0]

    @staticmethod
    def pack_u8(n: u8):
        return struct.pack('>B', n)

    @staticmethod
    def pack_i8(n: i8):
        return struct.pack('>b', n)

    @staticmethod
    def pack_u16(n: u16):
        return struct.pack('>H', n)

    @staticmethod
    def pack_i16(n: i16):
        return struct.pack('>h', n)

    @staticmethod
    def pack_u32(n: u32):
        return struct.pack('>I', n)

    @staticmethod
    def pack_i32(n: i32):
        return struct.pack('>i', n)

    @staticmethod
    def pack_u64(n: u64):
        return struct.pack('>Q', n)

    @staticmethod
    def pack_i64(n: i64):
        return struct.pack('>q', n)

    @staticmethod
    def pack_f32(n: f32):
        return struct.pack('>f', n)

    @staticmethod
    def pack_f64(n: f64):
        return struct.pack('>d', n)

class MemoryLoad:
    @staticmethod
    def i8(r: list):
        if utils.is_all_real(r[0]):
            return ctypes.c_int8(r[0]).value
        else:
            return r[0]

    @staticmethod
    def u8(r: list):
        return r[0]

    @staticmethod
    def i16(r: list):
        if utils.is_all_real(r[0], r[1]):
            return ctypes.c_int16(r[0] + r[1] * 2**8).value
        else:
            for i in range(len(r)):
                r[i] = utils.to_symbolic(r[i], 8)
            return z3.Concat(r[1], r[0])
            
    @staticmethod
    def u16(r: list):
        if utils.is_all_real(r[0], r[1]):
            return ctypes.c_uint16(r[0] + r[1] * 2**8).value
        else:
            for i in range(len(r)):
                r[i] = utils.to_symbolic(r[i], 8)
            return z3.Concat(r[1], r[0])

    @staticmethod
    def i32(r: list):
        if utils.is_all_real(r[0], r[1], r[2], r[3]):
            return ctypes.c_int32(r[0] + r[1] * 2**8 + r[2] * 2**16 + r[3] * 2**24).value
        else:
            for i in range(len(r)):
                r[i] = utils.to_symbolic(r[i], 8)
            return z3.Concat(r[3], r[2], r[1], r[0])

    @staticmethod
    def u32(r: list):
        if utils.is_all_real(r[0], r[1], r[2], r[3]):
            return ctypes.c_uint32(r[0] + r[1] * 2**8 + r[2] * 2**16 + r[3] * 2**24).value
        else:
            for i in range(len(r)):
                r[i] = utils.to_symbolic(r[i], 8)
            return z3.Concat(r[3], r[2], r[1], r[0])

    @staticmethod
    def i64(r: list):
        if utils.is_all_real(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]):
            return ctypes.c_int64(r[0] + r[1] * 2**8 + r[2] * 2**16 + r[3] * 2**24 + 
                                  r[4] * 2**32 + r[5] * 2**40 + r[6] * 2**48 + r[7] * 2**56)
        else:
            for i in range(len(r)):
                r[i] = utils.to_symbolic(r[i], 8)
            return z3.Concat(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0])
    @staticmethod
    def u64(r: list):
        if utils.is_all_real(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]):
            return ctypes.c_uint64(r[0] + r[1] * 2**8 + r[2] * 2**16 + r[3] * 2**24 + 
                                  r[4] * 2**32 + r[5] * 2**40 + r[6] * 2**48 + r[7] * 2**56)
        else:
            for i in range(len(r)):
                r[i] = utils.to_symbolic(r[i], 8)
            return z3.Concat(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0])
class MemoryStore:
    @staticmethod
    def pack_i8(n: i8):
        if utils.is_all_real(n):
            return [n & 0xFF]
        else:
            return [n]
    
    @staticmethod
    def pack_u8(n: u8):
        if utils.is_all_real(n):
            return [n & 0xFF]
        else:
            return [n]

    @staticmethod
    def pack_i16(n: i16):
        if utils.is_all_real(n):
            return [n & 0xFF, n & 0xFF00]
        else:
            return [z3.Extract(7, 0, n), z3.Extract(15, 8, n)]

    @staticmethod
    def pack_u16(n: u16):
        if utils.is_all_real(n):
            return [n & 0xFF, n & 0xFF00]
        else:
            return [z3.Extract(7, 0, n), z3.Extract(15, 8, n)]

    @staticmethod
    def pack_i32(n: i32):
        if utils.is_all_real(n):
            return [n & 0xFF, n & 0xFF00, n & 0xFF0000, n & 0xFF000000]
        else:
            return [z3.Extract(7, 0, n), z3.Extract(15, 8, n), z3.Extract(23, 16, n), z3.Extract(31, 24, n)]

    @staticmethod
    def pack_u32(n: u32):
        if utils.is_all_real(n):
            return [n & 0xFF, n & 0xFF00, n & 0xFF0000, n & 0xFF000000]
        else:
            return [z3.Extract(7, 0, n), z3.Extract(15, 8, n), z3.Extract(23, 16, n), z3.Extract(31, 24, n)]

    @staticmethod
    def pack_i64(n: i64):
        if utils.is_all_real(n):
            return [n & 0xFF, n & 0xFF00, n & 0xFF0000, n & 0xFF000000, n & 0xFF00000000,
                    n & 0xFF0000000000, n & 0xFF000000000000, n & 0xFF000000000000]
        else:
            return [z3.Extract(7, 0, n), z3.Extract(15, 8, n), z3.Extract(23, 16, n), z3.Extract(31, 24, n),
                    z3.Extract(39, 32, n), z3.Extract(47, 40, n), z3.Extract(55, 48, n), z3.Extract(63, 56, n)]
    @staticmethod
    def pack_u64(n: i64):
        if utils.is_all_real(n):
            return [n & 0xFF, n & 0xFF00, n & 0xFF0000, n & 0xFF000000, n & 0xFF00000000,
                    n & 0xFF0000000000, n & 0xFF000000000000, n & 0xFF000000000000]
        else:
            return [z3.Extract(7, 0, n), z3.Extract(15, 8, n), z3.Extract(23, 16, n), z3.Extract(31, 24, n),
                    z3.Extract(39, 32, n), z3.Extract(47, 40, n), z3.Extract(55, 48, n), z3.Extract(63, 56, n)]

def leb(reader, maxbits=32, signed=False):
    if isinstance(reader, (bytes, bytearray)):
        reader = io.BytesIO(reader)
    r = 0
    s = 0
    b = 0
    i = 0
    a = bytearray()
    while True:
        byte = ord(reader.read(1))
        i += 1
        a.append(byte)
        r |= ((byte & 0x7F) << s)
        s += 7
        if (byte & 0x80) == 0:
            break
        b += 1
        assert b <= math.ceil(maxbits / 7.0)
    if signed and (s < maxbits) and (byte & 0x40):
        r |= -(1 << s)
    return (i, r, a)

def rotl_u32(x: int, k: int):
    x = int2u32(x)
    k = int2u32(k)
    n = 32
    s = k & (n - 1)
    return (x << s | x >> (n - s)) & 0xFFFFFFFF

def rotl_u64(x: int, k: int):
    x = int2u64(x)
    k = int2u64(k)
    n = 64
    s = k & (n - 1)
    return (x << s | x >> (n - s)) & 0xFFFFFFFF

def rotr_u32(x: int, k: int):
    return rotl_u32(x, -k)

def rotr_u64(x: int, k: int):
    return rotl_u64(x, -k)

def idiv_s(a, b):
    return a // b if a * b > 0 else (a + (-a % b)) // b

def irem_s(a, b):
    return a % b if a * b > 0 else -(-a % b)