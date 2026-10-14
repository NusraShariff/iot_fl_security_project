import zlib

def compute_crc(data):
    return zlib.crc32(data.tobytes())