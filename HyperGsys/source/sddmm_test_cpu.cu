#include "../include/dataloader/dataloader.hpp"
#include "../include/hgnnAgg.cuh"
#include "../include/spgemm/spgemm.cuh"
#include "../include/spmm/spmm.cuh"
#include "../include/sddmm/sddmm.cuh"
#include "../include/util/ramArray.cuh"

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpGEMM
#include <fstream>
#include <stdio.h>  // printf
#include <stdlib.h> // EXIT_FAILURE

__global__ void warm_up() {}

// template <class DType>
// __global__ void resetH(int64_t nnz, DType *A, DType *B)
// {
//   int tid = blockDim.x * blockIdx.x + threadIdx.x;
//   int total = blockDim.x * gridDim.x;
//   for (int64_t i = tid; i < nnz; i += total)
//   {
//     A[i] = B[i];
//   }
// }

int main(int argc, char **argv)
{
  // Host problem definition
  if (argc < 3)
  {
    printf("Input: first get the path of sparse matrix, then get the "
           "feature length of dense matrix\n");
    exit(1);
  }
  char *filename = argv[1];
  int feature_size = atoi(argv[2]);
  int max_load = atoi(argv[3]);

  const int iter = 300;
  auto SpPair = DataLoader<Index, DType>(filename);

  std::fstream fs;
  fs.open("sddmm_result.csv", std::ios::app | std::ios::in | std::ios::out);
  // fs.open("result.csv", std::ios::in | std::ios::out);
  // fs << "dataset"
  //    << ","
  //    << "feature size"
  //    << ","
  //    << "max_load"
  //    << ","
  //    << "cusparse"
  //    << ","
  //    << "csrscale"
  //    << ","
  //    << "edge group"
  //    << ","
  //    << "\n";

  // SpMatCsrDescr_t<Index, DType> H = std::get<0>(SpPair);

  // util::RamArray<DType> in_feature1(H.nrow * feature_size);
  // util::RamArray<DType> in_feature2(H.ncol * feature_size);
  // util::RamArray<DType> out_feature(H.nnz);
  // util::RamArray<DType> out_ref(H.nnz);

  // gnn_balancer<Index, DType, balan_met::gnn_edge_group> balan_edgegroup(max_load, H);

  // // in_feature.fill_default_one();
  // in_feature1.fill_random_h();
  // in_feature2.fill_random_h();
  // // in_feature1.fill_default_one();
  // // in_feature2.fill_default_one();
  // out_feature.fill_zero_h();
  // out_ref.fill_zero_h();
  // in_feature1.upload();
  // in_feature2.upload();
  // out_feature.upload();
  // H.upload();

  // printf("start sddmm test\n");
  // // warm up

  // for (int i = 0; i < 1000; i++)
  //   warm_up<<<1, 1>>>();

  // fs << filename << "," << feature_size << "," << max_load << ",";

  // if (SDDMM_check<Index, DType, sddmm_kernel_met::sddmm_cusparse>(
  //         feature_size, H, in_feature1, in_feature2, out_ref))

  //   SDDMM_test<Index, DType, sddmm_kernel_met::sddmm_cusparse>(
  //       fs, iter, feature_size, H, in_feature1, in_feature2);

  // if (SDDMM_check<Index, DType, sddmm_kernel_met::sddmm_csrscale>(
  //         feature_size, H, in_feature1, in_feature2, out_ref))
  //   SDDMM_test<Index, DType, sddmm_kernel_met::sddmm_csrscale>(
  //       fs, iter, feature_size, H, in_feature1, in_feature2);

  // if (SDDMM_check<Index, DType, sddmm_kernel_met::sddmm_edge_group, balan_met::gnn_edge_group>(
  //         feature_size, H, balan_edgegroup, in_feature1, in_feature2, out_ref))

  //   SDDMM_test<Index, DType, sddmm_kernel_met::sddmm_edge_group, balan_met::gnn_edge_group>(
  //       fs, iter, feature_size, H, balan_edgegroup, in_feature1, in_feature2);
  fs << filename << "," << feature_size << "," << max_load << ",";

  for (sddmm_alg = sddmm_cusparse; sddmm_alg <= sddmm_edge_group; sddmm_alg = (sddmm_kernel_met)(sddmm_alg + 1))
  {
    SpMatCsrDescr_t<Index, DType> H = std::get<0>(SpPair);

    util::RamArray<DType> in_feature1(H.nrow * feature_size);
    util::RamArray<DType> in_feature2(H.ncol * feature_size);
    util::RamArray<DType> out_feature(H.nnz);
    util::RamArray<DType> out_ref(H.nnz);

    gnn_balancer<Index, DType, balan_met::gnn_edge_group> balan_edgegroup(max_load, H);

    // in_feature.fill_default_one();
    in_feature1.fill_random_h();
    in_feature2.fill_random_h();
    // in_feature1.fill_default_one();
    // in_feature2.fill_default_one();
    out_feature.fill_zero_h();
    out_ref.fill_zero_h();
    in_feature1.upload();
    in_feature2.upload();
    out_feature.upload();
    H.upload();

    printf("start sddmm test\n");
    // warm up

    for (int i = 0; i < 1000; i++)
      warm_up<<<1, 1>>>();

    switch (sddmm_alg)
    {
    case sddmm_cusparse:
      /* code */
      if (SDDMM_check<Index, DType, sddmm_kernel_met::sddmm_cusparse>(
              feature_size, H, in_feature1, in_feature2, out_ref))

        SDDMM_test<Index, DType, sddmm_kernel_met::sddmm_cusparse>(
            fs, iter, feature_size, H, in_feature1, in_feature2);
      break;
    case sddmm_csrscale:
      /* code */
      if (SDDMM_check<Index, DType, sddmm_kernel_met::sddmm_csrscale>(
              feature_size, H, in_feature1, in_feature2, out_ref))
        SDDMM_test<Index, DType, sddmm_kernel_met::sddmm_csrscale>(
            fs, iter, feature_size, H, in_feature1, in_feature2);
      break;
    case sddmm_edge_group:
      /* code */
      if (SDDMM_check<Index, DType, sddmm_kernel_met::sddmm_edge_group, balan_met::gnn_edge_group>(
              feature_size, H, balan_edgegroup, in_feature1, in_feature2, out_ref))

        SDDMM_test<Index, DType, sddmm_kernel_met::sddmm_edge_group, balan_met::gnn_edge_group>(
            fs, iter, feature_size, H, balan_edgegroup, in_feature1, in_feature2);
      break;
    default:
      break;
    }
    if (sddmm_alg == sddmm_edge_group)
      fs << "\n";
  }
  fs.close();
  return 0;
}