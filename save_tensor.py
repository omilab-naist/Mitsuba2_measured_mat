import numpy as np 
import struct
def saveTensor(filename, dict_data):

    #Windows 10ではリトルエンディアン'<'での動作を確認
    ByteOrder = '<'

    enumType = {
        np.dtype('uint8'): 1,    np.dtype('int8'): 2,
        np.dtype('uint16'): 3,   np.dtype('int16'): 4,
        np.dtype('uint32'): 5,   np.dtype('int32'): 6,
        np.dtype('uint64'): 7,   np.dtype('int64'): 8,
        np.dtype('float16'): 9,  np.dtype('float32'): 10,  np.dtype('float64'): 11,
    }
    with open(filename, 'wb') as fout:
        # header
        fout.write(b'tensor_file\0')
        fout.write(b'X\0')  # version
        fout.write(struct.pack('L', len(dict_data)))
        # make data header
        dataheader = []
        datapos = fout.tell()
        for i, (key, val) in enumerate(sorted(dict_data.items())):
            b = struct.pack('%cH'%ByteOrder, len(key))  # name_length
            b += key.encode()                # name
            b += struct.pack('%cHBQ'%ByteOrder, len(val.shape), enumType[val.dtype], 0)  # ndim, dtype, offset
            b += struct.pack('%c%dQ'%(ByteOrder,len(val.shape)), *val.shape)  # shape
            dataheader.append(b)
        datapos += np.sum([len(b) for b in dataheader])
        # write data header
        for i, (key, val) in enumerate(sorted(dict_data.items())):
            b = struct.pack('%cH'%ByteOrder, len(key))  # name_length
            b += key.encode()                # name
            b += struct.pack('%cHBQ'%ByteOrder, len(val.shape), enumType[val.dtype], datapos)  # ndim, dtype, offset
            b += struct.pack('%c%dQ'%(ByteOrder,len(val.shape)), *val.shape)  # shape
            fout.write(b)
            datapos += val.nbytes
        for i, (key, val) in enumerate(sorted(dict_data.items())):
            fout.write(val.tobytes())