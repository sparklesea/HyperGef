import os
import argparse

parser = argparse.ArgumentParser(description='命令行中传入kernel name')
# type是要传入的参数的数据类型  help是该参数的提示信息
parser.add_argument('--kernel_name', "-k", type=str, help='传入的kernel name')
args = parser.parse_args()

path = "../data/mtx"  # SuiteSparse path
for root, dirs, files in os.walk(path):
    print(files)
    files[:] = [f for f in files if (f.endswith(".mtx"))]
    for filename in files:
        pathmtx = os.path.join(path, filename)
        if args.kernel_name == 'spmm':
            cmd = "./spmm_test_cpu %s %d %d" % (pathmtx, 32, 8)
        elif args.kernel_name == 'sddmm':
            cmd = "./sddmm_test_cpu %s %d %d" % (pathmtx, 32, 8)
        print(cmd)
        os.system(cmd)
