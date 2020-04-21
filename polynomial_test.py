import bchlib
import hashlib
import os
import random

# create a bch object
for BCH_POLYNOMIAL in range(200, 300):  # 尝试生成多项式的十进制数
    try:
        BCH_BITS = 4
        bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

        # random data
        data = bytearray(os.urandom(12))
        print(data)
        # encode and make a "packet"
        ecc = bch.encode(data)
        packet = data + ecc
        print(ecc)
        print(len(ecc))
        # print hash of packet
        sha1_initial = hashlib.sha1(packet)
        print('sha1: %s' % (sha1_initial.hexdigest(), ))

        def bitflip(packet):
            byte_num = random.randint(0, len(packet) - 1)
            bit_num = random.randint(0, 7)
            packet[byte_num] ^= (1 << bit_num)

        # make BCH_BITS errors
        for _ in range(BCH_BITS):
            bitflip(packet)

        # print hash of packet
        sha1_corrupt = hashlib.sha1(packet)
        print('sha1: %s' % (sha1_corrupt.hexdigest(), ))

        # de-packetize
        data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]
        # correct
        bitflips = bch.decode_inplace(data, ecc)
        print('bitflips: %d' % (bitflips))

        # packetize
        packet = data + ecc

        # print hash of packet
        sha1_corrected = hashlib.sha1(packet)
        print('sha1: %s' % (sha1_corrected.hexdigest(), ))

        if sha1_initial.digest() == sha1_corrected.digest():
            print('Corrected!')
        else:
            print('Failed')
        print(BCH_POLYNOMIAL)
        print(bch.m)  # gf域
        print(bch.n)  # BCH能编码的最大信息长度
        print(bch.ecc_bits)  # ecc所占位数
        print(bch.t)  # BCH能纠正的位数
        print("----------------")
    except Exception as err:
        print(err)