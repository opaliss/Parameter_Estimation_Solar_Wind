import math
from multiprocess import Pool, cpu_count
import time
from functools import partial


def check_prime(Num):
    arr = [True] * Num
    for num in range(Num):
        if num < 2:
            arr[num] = False
        elif num > 2:
            for j in range(2, math.ceil(math.sqrt(num)) + 1):
                if num % j == 0:
                    arr[num] = False
                    break
    return arr


def check_prime_multi(num, other_var):
    if num < 2:
        return num, False
    elif num == 2:
        return num, True
    else:
        for j in range(2, math.ceil(math.sqrt(num)) + 1):
            if num % j == 0:
                return num, False
    return num, True


if __name__ == "__main__":
    # results from running on poseidon:
    # (tutorial-env) oissan@poseidon:~/UQ_GONG/multi_processing$ python -m multi_tutorial2

    # setting N= 10^7 we get:
    # time taken single-processing:  130.34289240837097 s
    # time taken multi-processing: 33.13105845451355 s

    # setting N=10^8 we get:
    # time taken single-processing: 2046.249838590622 s
    # time taken multi-processing: 169.46101880073547 s

    # 12 times faster to run using multiprocessing!!!
    N = 10 ** 7

    # single-processing
    start_time = time.time()
    results = check_prime(N)
    print(results[:30])
    print("time taken single-processing:", time.time() - start_time)

    # multi-processing
    # this is the most elegant way to write parallel code in Python!!
    start_time = time.time()

    # [0, 1, 2, ...., N-1]
    num_arr = range(N)

    # keep one cpu empty? recommended online...
    num_process = cpu_count() - 1
    print("cpu count  = ", cpu_count())
    with Pool(processes=num_process) as pool:
        # pool map takes the function we want to iterate and runs the given function for each iterate (not in order).
        # parallelize a single job!
        results2 = pool.map(partial(check_prime_multi, other_var=0), num_arr)

    # close process when done.
    pool.close()

    # print result of first 30 numbers.
    print(results2[:30])
    print("time taken multi-processing:", time.time() - start_time)
