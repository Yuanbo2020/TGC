import os
from time import sleep
from multiprocessing import Pool
import multiprocessing as mp

gpu_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

file = '1_main.py'
filepath = os.path.join(os.getcwd(), file)

times = 1

batch_size = 64

lr_init_list = [1e-4]

epochs = 100

check_step = 1

window_list = [144]

layers = [6]


def run_jobs():
    for sub_time in range(times):
        each_time = sub_time
        for lr_init in lr_init_list:
            for layer in layers:
                for window in window_list:
                    command = 'python {0}  ' \
                              '   -times {1} ' \
                              '-batch_size {2} -lr_init {3} ' \
                              ' -epochs {4}  -check_step {5} ' \
                              ' -window {6} ' \
                              ' -encoder_layer {7}   ' \
                              ''.format(filepath,
                                        each_time,
                                        batch_size,
                                        lr_init,
                                        epochs, check_step,
                                        window,
                                        layer)

                    print(command)
                    if os.system(command):
                        print('\nFailed: ', command, '\n')
                        sleep(1)


def main():
    cpu_num = mp.cpu_count()
    print('cpu_num: ', cpu_num)

    cpu_num = 1
    pool = Pool(cpu_num)

    if cpu_num == 1:
        pool.apply_async(func=run_jobs, args=())
    else:
        for i in range(cpu_num * times * len(lr_init_list)):
            pool.apply_async(func=run_jobs, args=())

    pool.close()
    pool.join()



if __name__ == '__main__':
    main()

