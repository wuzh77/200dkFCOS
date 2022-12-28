import numpy as np

if __name__ == "__main__":
    # binPath = '/home/dongyu3/wuzhFCOS/binresult/bboxres139.bin'
    # bboxes = np.fromfile(binPath, dtype=np.float32)
    # bboxes = np.reshape(bboxes, [100, 5])
    # bboxes.tolist()
    # print(bboxes)
    # [[3.92718744e+00 1.65607498e+02 1.54957504e+02 2.61590637e+02
    #   5.97167969e-01]
    #  [2.92608765e+02 2.16328125e+02 3.51982513e+02 3.18168762e+02
    #   5.64453125e-01]
    # [ 5.49148132e+02  2.95399048e+02  5.85580261e+02  4.01004456e+02
    #    3.63525391e-01]

    # [[ 4.00873327e+00  1.67135773e+02  1.52805145e+02  2.61510773e+02
    #    6.14746094e-01]
    #  [ 2.92589386e+02  2.16195007e+02  3.52798218e+02  3.14707520e+02
    #    4.65087891e-01]
    # [3.60502502e+02 2.19656250e+02 4.28396240e+02 3.16571259e+02
    #   4.61425781e-01]
    labelPath = '/home/dongyu3/MxBaseFCOS/binresult/000000000139class.bin'
    labels = np.fromfile(labelPath, dtype=np.int64)
    labels = np.reshape(labels, [100, 1])
    print(labels)

