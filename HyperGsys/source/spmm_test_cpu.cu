#include "../include/dataloader/dataloader.hpp"
#include "../include/hgnnAgg.cuh"
#include "../include/spgemm/spgemm.cuh"
#include "../include/spmm/spmm.cuh"
#include "../include/util/ramArray.cuh"

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpGEMM
#include <fstream>
#include <stdio.h>  // printf
#include <stdlib.h> // EXIT_FAILURE

__global__ void warm_up() {}

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
  fs.open("spmm_result.csv", std::ios::app | std::ios::in | std::ios::out);
  // fs.open("result.csv", std::ios::in | std::ios::out);
  fs << "dataset"
     << ","
     << "feature size"
     << ","
     << "cusparse"
     << ","
     << "row balance"
     << ","
     << "edge balance"
     << ","
     << "edge group"
     << ","
     << "\n";
  SpMatCsrDescr_t<Index, DType> H = std::get<0>(SpPair);

  util::RamArray<DType> in_feature(H.ncol * feature_size);
  util::RamArray<DType> out_feature(H.nrow * feature_size);
  util::RamArray<DType> out_ref(H.nrow * feature_size);

  gnn_balancer<Index, DType, balan_met::gnn_edge_group> balan_edgegroup(max_load, H);

  // in_feature.fill_default_one();
  in_feature.fill_random_h();
  out_feature.fill_zero_h();
  in_feature.upload();
  out_feature.upload();
  H.upload();

  printf("start spmm test\n");
  // warm up
  for (int i = 0; i < 1000; i++)
    warm_up<<<1, 1>>>();

  fs << filename << "," << feature_size << "," << max_load << ",";
  if (SpMM_check<Index, DType, spmm_kernel_met::cusparse>(
          feature_size, H, in_feature, out_feature, out_ref))

    SpMM_test<Index, DType, spmm_kernel_met::cusparse>(
        fs, iter, feature_size, H, in_feature, out_feature);

  if (SpMM_check<Index, DType, spmm_kernel_met::row_balance>(
          feature_size, H, in_feature, out_feature, out_ref))

    SpMM_test<Index, DType, spmm_kernel_met::row_balance>(
        fs, iter, feature_size, H, in_feature, out_feature);

  if (SpMM_check<Index, DType, spmm_kernel_met::edge_balance>(
          feature_size, H, in_feature, out_feature, out_ref))

    SpMM_test<Index, DType, spmm_kernel_met::edge_balance>(
        fs, iter, feature_size, H, in_feature, out_feature);

  if (SpMM_check<Index, DType, spmm_kernel_met::edge_group, balan_met::gnn_edge_group>(
          feature_size, H, balan_edgegroup, in_feature, out_feature, out_ref))

    SpMM_test<Index, DType, spmm_kernel_met::edge_group, balan_met::gnn_edge_group>(
        fs, iter, feature_size, H, balan_edgegroup, in_feature, out_feature);

  fs << "\n";
  fs.close();
  return 0;
}