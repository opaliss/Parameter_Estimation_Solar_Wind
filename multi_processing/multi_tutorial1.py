from multiprocess import Process
import numpy as np
import time


# counts 0, 1, 2, 3, ..., N
def counter1(num):
    cnt = 0
    for _ in range(num):
        cnt += 1
    print("counter1 done!")


# counts 0, 2, 4, ..., N
def counter2(num):
    cnt = 0
    for _ in range(0, num, 2):
        cnt += 1
    print("counter2 done!")


if __name__ == "__main__":
    # note, count2 is faster than count1 so when running multiprocessing you should see that count2 finished
    # before count1!
    N = 10 ** 7

    # single-processing
    start_time = time.time()
    counter1(N)
    counter2(N)
    print("total time single-processing = ", time.time() - start_time)

    # multi-processing
    start_time = time.time()
    # initialize process
    p1 = Process(target=counter1, args=(N,))
    p2 = Process(target=counter2, args=(N,))

    # start the processes
    p1.start()
    p2.start()

    # join the processes
    p1.join()
    p2.join()
    print("total time multi-processing = ", time.time() - start_time)
