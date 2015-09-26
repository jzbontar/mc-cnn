import numpy as np

left = np.memmap('../left.bin', dtype=np.float32, shape=(1, 70, 370, 1226))
right = np.memmap('../right.bin', dtype=np.float32, shape=(1, 70, 370, 1226))
disp = np.memmap('../disp.bin', dtype=np.float32, shape=(1, 1, 370, 1226))
