import os
path = "../data/mtx"  # SuiteSparse path
for (root, dirs, files) in os.walk(path):
    print(files)
    files[:] = [f for f in files if (f.endswith(".mtx"))]
    for max_load in [4, 8, 16, 32, 64]:
        for filename in files:
            pathmtx = os.path.join(path, filename)
            cmd = "./spmm_test_cpu %s %d %d" % (pathmtx, 32, max_load)
            print(cmd)
            os.system(cmd)