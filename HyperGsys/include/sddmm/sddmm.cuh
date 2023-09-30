#ifndef SDDMM_
#define SDDMM_

#include "../dataloader/dataloader.hpp"
#include "../taskbalancer/balancer.cuh"
#include "../util/check.cuh"
#include "../util/gpuTimer.cuh"
#include "../util/ramArray.cuh"
#include <cuda.h>
#include <cusparse.h>
#include <fstream>
#include <string>

enum sddmm_kernel_met
{
    sddmm_cusparse,
    sddmm_csrscale,
    sddmm_edge_group,
}sddmm_alg;

template <typename Index, typename DType>
void csrsddmm_cusparse(const int nrow, const int ncol, const int nnz,
                       const int feature_size, int *sp_csrptr, int *sp_csrind,
                       DType *sp_data, DType *in_feature1, DType *in_feature2)
{
    //
    // Run Cusparse-SDDMM and check result
    //
    cusparseHandle_t handle;
    cusparseSpMatDescr_t csrInputDescr;
    cusparseDnMatDescr_t dnMatInputDescr1, dnMatInputDescr2;
    float alpha = 1.0f, beta = 0.0f;

    checkCuSparseError(cusparseCreate(&handle));

    // creating sparse csr matrix
    checkCuSparseError(cusparseCreateCsr(
        &csrInputDescr, nrow, ncol, nnz, sp_csrptr, sp_csrind, sp_data,
        CUSPARSE_INDEX_32I, // index 32-integer for indptr
        CUSPARSE_INDEX_32I, // index 32-integer for indices
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F // datatype: 32-bit float real number
        ));

    // creating dense matrices
    checkCuSparseError(cusparseCreateDnMat(&dnMatInputDescr1, nrow, feature_size,
                                           feature_size, in_feature1, CUDA_R_32F,
                                           CUSPARSE_ORDER_ROW));
    checkCuSparseError(cusparseCreateDnMat(&dnMatInputDescr2, ncol, feature_size,
                                           feature_size, in_feature2, CUDA_R_32F,
                                           CUSPARSE_ORDER_ROW));

    // allocate workspace buffer
    size_t workspace_size;
    checkCuSparseError(cusparseSDDMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_TRANSPOSE, &alpha, dnMatInputDescr1, dnMatInputDescr2,
        &beta, csrInputDescr, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT,
        &workspace_size));

    void *workspace = NULL;
    checkCudaError(cudaMalloc(&workspace, workspace_size));

    // run SDDMM
    checkCuSparseError(cusparseSDDMM(handle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                                     CUSPARSE_OPERATION_TRANSPOSE,     // opB
                                     &alpha, dnMatInputDescr1, dnMatInputDescr2, &beta,
                                     csrInputDescr, CUDA_R_32F,
                                     CUSPARSE_SDDMM_ALG_DEFAULT, workspace));

    checkCuSparseError(cusparseDestroy(handle));
    checkCudaError(cudaFree(workspace));
    checkCuSparseError(cusparseDestroySpMat(csrInputDescr));

    checkCuSparseError(cusparseDestroyDnMat(dnMatInputDescr1));
    checkCuSparseError(cusparseDestroyDnMat(dnMatInputDescr2));
}

template <typename Index, typename DType>
float csrsddmm_cusparse_test(int iter, SpMatCsrDescr_t<Index, DType> &H,
                             const Index feature_size, DType *in_feature1,
                             DType *in_feature2)
{
    cusparseHandle_t handle;
    cusparseSpMatDescr_t csrInputDescr;
    cusparseDnMatDescr_t dnMatInputDescr1, dnMatInputDescr2;
    float alpha = 1.0f, beta = 0.0f;

    checkCuSparseError(cusparseCreate(&handle));

    // creating sparse csr matrix
    checkCuSparseError(cusparseCreateCsr(
        &csrInputDescr, H.nrow, H.ncol, H.nnz, H.sp_csrptr.d_array.get(), H.sp_csrind.d_array.get(), H.sp_data.d_array.get(),
        CUSPARSE_INDEX_32I, // index 32-integer for indptr
        CUSPARSE_INDEX_32I, // index 32-integer for indices
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F // datatype: 32-bit float real number
        ));

    // creating dense matrices
    checkCuSparseError(cusparseCreateDnMat(&dnMatInputDescr1, H.nrow, feature_size,
                                           feature_size, in_feature1, CUDA_R_32F,
                                           CUSPARSE_ORDER_ROW));
    checkCuSparseError(cusparseCreateDnMat(&dnMatInputDescr2, H.ncol, feature_size,
                                           feature_size, in_feature2, CUDA_R_32F,
                                           CUSPARSE_ORDER_ROW));

    // allocate workspace buffer
    size_t workspace_size;
    checkCuSparseError(cusparseSDDMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_TRANSPOSE, &alpha, dnMatInputDescr1, dnMatInputDescr2,
        &beta, csrInputDescr, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT,
        &workspace_size));

    void *workspace = NULL;
    checkCudaError(cudaMalloc(&workspace, workspace_size));

    // run SDDMM
    util::gpuTimer atimer;
    atimer.start();
    for (int i = 0; i < iter; i++)
        checkCuSparseError(cusparseSDDMM(handle,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                                         CUSPARSE_OPERATION_TRANSPOSE,     // opB
                                         &alpha, dnMatInputDescr1, dnMatInputDescr2, &beta,
                                         csrInputDescr, CUDA_R_32F,
                                         CUSPARSE_SDDMM_ALG_DEFAULT, workspace));
    atimer.end();

    checkCuSparseError(cusparseDestroy(handle));
    checkCudaError(cudaFree(workspace));
    checkCuSparseError(cusparseDestroySpMat(csrInputDescr));

    checkCuSparseError(cusparseDestroyDnMat(dnMatInputDescr1));
    checkCuSparseError(cusparseDestroyDnMat(dnMatInputDescr2));

    return atimer.elapsed();
}

__global__ void sddmmCSR2Scale_kernel(const int S_mrows, int D_kcols,
                                      const unsigned long Size, int *S_csrRowPtr,
                                      int *S_csrColInd, float *D1_dnVal,
                                      float *D2_dnVal, float *O_csrVal)
{
    int eid = (blockIdx.x << 4) + (threadIdx.y << 2);
    int cid = threadIdx.x << 1;

    if (blockIdx.x < Size / 16)
    {
        float multi[4] = {0, 0, 0, 0};
        int offset1[4], offset2[4];
        float2 D1tmp[4], D2tmp[4];
        Load<int4, int>(offset2, S_csrColInd, eid);
        offset1[0] = findRow(S_csrRowPtr, eid, 0, S_mrows);
        offset1[3] = findRow(S_csrRowPtr, eid + 3, offset1[0], S_mrows);
        offset1[1] = findRow(S_csrRowPtr, eid + 1, offset1[0], offset1[3]);
        offset1[2] = findRow(S_csrRowPtr, eid + 2, offset1[1], offset1[3]);
        selfMulConst4<int>(offset1, D_kcols);
        selfMulConst4<int>(offset2, D_kcols);

        for (int i = 0; i < (D_kcols >> 5); i++)
        {
            Load4<float2, float>(D1tmp, D1_dnVal, offset1, cid);
            Load4<float2, float>(D2tmp, D2_dnVal, offset2, cid);
            vec2Dot4<float2>(multi, D1tmp, D2tmp);
            cid += 32;
        }
        int res = D_kcols & 31;
        if (res)
        {
            int cid2 = threadIdx.x + D_kcols - res;
            float D1[4] = {0, 0, 0, 0}, D2[4] = {0, 0, 0, 0};
            for (int i = 0; i < (res >> 4) + 1; i++)
            {
                if ((i << 4) + threadIdx.x < res)
                {
                    Load4<float, float>(D1, D1_dnVal, offset1, cid2);
                    Load4<float, float>(D2, D2_dnVal, offset2, cid2);
                    Dot4<float>(multi, D1, D2);
                    cid2 += 16;
                }
            }
        }
        AllReduce4<float>(multi, 8, 32);

        if (threadIdx.x == 0)
        {
            for (int i = 0; i < 4; ++i)
            {
                multi[i] *= O_csrVal[eid + i];
            }
            Store<float4, float>(O_csrVal, multi, eid);
        }
    }
    else // Dynamic parrallel?
    {
        eid = Size - (Size & 15) + (blockIdx.x - (Size / 16));
        int offset1 = findRow(S_csrRowPtr, eid, 0, S_mrows) * D_kcols;
        int offset2 = S_csrColInd[eid] * D_kcols;
        float multi = 0;
        int off1 = cid = (threadIdx.y << 4) + threadIdx.x;
        float D1tmp0, D2tmp0;
        for (int cc = 0; cc < (D_kcols >> 5); cc++)
        {
            D1tmp0 = D1_dnVal[offset1 + cid];
            D2tmp0 = D2_dnVal[offset2 + cid];
            multi += D1tmp0 * D2tmp0;
            cid += 32;
        }
        int res = D_kcols & 31;
        D1tmp0 = D2tmp0 = 0;
        if (res)
        {
            if (off1 < res)
            {
                D1tmp0 = D1_dnVal[offset1 + cid];
                D2tmp0 = D2_dnVal[offset2 + cid];
            }
            multi += D1tmp0 * D2tmp0;
        }
        for (int stride = 16; stride > 0; stride >>= 1)
        {
            multi += __shfl_xor_sync(0xffffffff, multi, stride, 32);
        }
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            O_csrVal[eid] *= multi;
        }
    }
}

__global__ void sddmmCSR1Scale_kernel(const int S_mrows, int D_kcols,
                                      const unsigned long Size, int *S_csrRowPtr,
                                      int *S_csrColInd, float *D1_dnVal,
                                      float *D2_dnVal, float *O_csrVal)
{
    int eid = (blockIdx.x << 4) + (threadIdx.y << 2);
    int cid = threadIdx.x;

    if (blockIdx.x < Size / 16)
    {
        float multi[4] = {0, 0, 0, 0};
        int offset1[4], offset2[4];
        float D1tmp[4], D2tmp[4];
        int length[4] = {1, 1, 1, 1};

        Load<int4, int>(offset2, S_csrColInd, eid);

        offset1[0] = findRow(S_csrRowPtr, eid, 0, S_mrows);
        offset1[3] = findRow(S_csrRowPtr, eid + 3, offset1[0], S_mrows);
        offset1[1] = findRow(S_csrRowPtr, eid + 1, offset1[0], offset1[3]);
        offset1[2] = findRow(S_csrRowPtr, eid + 2, offset1[1], offset1[3]);

        for (int i = 0; i < 4; i++)
        {
            length[i] = S_csrRowPtr[offset1[i] + 1] - S_csrRowPtr[offset1[i]];
        }
        selfMulConst4<int>(offset1, D_kcols);
        selfMulConst4<int>(offset2, D_kcols);

        for (int i = 0; i < (D_kcols >> 5); i++)
        {
            Load4<float, float>(D1tmp, D1_dnVal, offset1, cid);
            Load4<float, float>(D2tmp, D2_dnVal, offset2, cid);
            Dot4<float>(multi, D1tmp, D2tmp);
            cid += 32;
        }
        int res = D_kcols & 31;
        if (res)
        {
            float D1[4] = {0, 0, 0, 0}, D2[4] = {0, 0, 0, 0};
            if (threadIdx.x < res)
            {
                Load4<float, float>(D1, D1_dnVal, offset1, cid);
                Load4<float, float>(D2, D2_dnVal, offset2, cid);
                Dot4<float>(multi, D1, D2);
            }
        }
        AllReduce4<float>(multi, 16, 32);
        if (threadIdx.x == 0)
        {
            for (int i = 0; i < 4; ++i)
            {
                multi[i] *= O_csrVal[eid + i];
            }
            Store<float4, float>(O_csrVal, multi, eid);
        }
    }
    else // Dynamic parrallel?
    {
        eid = Size - (Size & 15) + (blockIdx.x - (Size / 16));
        int offset1 = findRow(S_csrRowPtr, eid, 0, S_mrows) * D_kcols;
        int length =
            S_csrRowPtr[offset1 / D_kcols + 1] - S_csrRowPtr[offset1 / D_kcols];
        int offset2 = S_csrColInd[eid] * D_kcols;
        float multi = 0;
        int off1 = cid = threadIdx.x;
        float D1tmp0, D2tmp0;
        for (int cc = 0; cc < (D_kcols >> 5); cc++)
        {
            D1tmp0 = D1_dnVal[offset1 + cid];
            D2tmp0 = D2_dnVal[offset2 + cid];
            multi += D1tmp0 * D2tmp0;
            cid += 32;
        }
        int res = D_kcols & 31;
        D1tmp0 = D2tmp0 = 0;
        if (res)
        {
            if (off1 < res)
            {
                D1tmp0 = D1_dnVal[offset1 + cid];
                D2tmp0 = D2_dnVal[offset2 + cid];
            }
            multi += D1tmp0 * D2tmp0;
        }
        for (int stride = 16; stride > 0; stride >>= 1)
        {
            multi += __shfl_xor_sync(0xffffffff, multi, stride, 32);
        }
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            O_csrVal[eid] *= multi;
        }
    }
}

template <typename Index, typename DType>
void csrsddmm_scale(SpMatCsrDescr_t<Index, DType> &spmatA,
                    const Index feature_size,
                    DType *D1, DType *D2)
{
    const auto m = spmatA.nrow;
    const auto k = feature_size;
    const auto nnz = spmatA.nnz;
    if ((k % 2) == 0)
    {
        sddmmCSR2Scale_kernel<<<dim3(nnz / 16 + (nnz & 15), 1, 1), dim3(16, 4, 1)>>>(
            m, k, nnz, spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(), D1, D2, spmatA.sp_data.d_array.get());
    }
    else
    {
        sddmmCSR1Scale_kernel<<<dim3(nnz / 16 + (nnz & 15), 1, 1), dim3(32, 4, 1)>>>(m, k, nnz, spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(), D1, D2, spmatA.sp_data.d_array.get());
    }
}

template <typename Index, typename DType>
__global__ void
csrsddmm_edgegroup_kernel(const Index edge_groups, const Index feature_size,
                          const Index group_key[], const Index group_row[],
                          const Index colIdx[], DType values[],
                          const DType D1[], const DType D2[])
{
    Index group_tile = blockDim.y; // combine a set of groups together
    Index subwarp_id = threadIdx.y;
    Index group = blockIdx.x * group_tile + subwarp_id; // which node_group
    Index v_id = threadIdx.x;
    if (group < edge_groups)
    {
        Index row = group_row[group]; // get the specific row of each node group
        // D1 += v_id;
        // D2 += v_id;
        // dnOutput += v_id;
        DType res, val;
        Index col;
        Index start = __ldg(group_key + group);
        Index end = __ldg(group_key + group + 1);
        for (Index p = start; p < end; p++)
        {
            res = 0;
            col = __ldg(colIdx + p);
            val = util::__guard_load_default_one<DType>(values, p);
            for (Index k = 0; k < feature_size; ++k)
            {
                res += __ldg(D1 + row * feature_size + k) * __ldg(D2 + col * feature_size + k);
            }
            values[p] = res * val;
            // dnOutput[p] = res * val;
        }
        // atomicAdd(dnOutput + row * feature_size,res); // atomic, cuz different node group -> same row
    }
}

template <typename Index, typename DType>
void csrsddmm_edgegroup(SpMatCsrDescr_t<Index, DType> &spmatA,
                        const Index feature_size, const Index edge_groups,
                        const Index *group_key, const Index *group_row,
                        const DType *D1, const DType *D2)
{
    Index Mdim_worker = edge_groups;
    Index Ndim_worker = feature_size;

    Index ref_block = (feature_size > 256) ? feature_size : 256;
    Index Ndim_threadblock = CEIL(Ndim_worker, ref_block);
    Index Ndim_thread_per_tb = min(Ndim_worker, ref_block);
    Index Mdim_thread_per_tb = CEIL(ref_block, Ndim_thread_per_tb);
    Index Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);
    // size_t shr_size = feature_size * Mdim_thread_per_tb * sizeof(DType);

    dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
    dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

    csrsddmm_edgegroup_kernel<<<gridDim, blockDim>>>(
        edge_groups, feature_size, group_key, group_row,
        spmatA.sp_csrind.d_array.get(), spmatA.sp_data.d_array.get(), D1, D2);
}

template <class Index, class DType>
void SDDMM_host(int feature_size, SpMatCsrDescr_t<Index, DType> &H,
                util::RamArray<DType> &D1,
                util::RamArray<DType> &D2,
                util::RamArray<DType> &out_ref)
{
    out_ref.reset();
    util::sddmm_reference_host<Index, DType>(
        H.nrow, H.ncol, feature_size, H.sp_csrptr.h_array.get(),
        H.sp_csrind.h_array.get(), H.sp_data.h_array.get(),
        D1.h_array.get(), D2.h_array.get(), out_ref.h_array.get());
}

// check device based on ref
template <class Index, class DType, sddmm_kernel_met km, balan_met bm>
bool SDDMM_check(int feature_size, SpMatCsrDescr_t<Index, DType> &H,
                 gnn_balancer<Index, DType, bm> &balan,
                 util::RamArray<DType> &D1,
                 util::RamArray<DType> &D2,
                 util::RamArray<DType> &out_ref)
{
    out_ref.reset();
    SDDMM_host<Index, DType>(feature_size, H, D1, D2, out_ref);
    if (km == sddmm_kernel_met::sddmm_edge_group)
    {
        csrsddmm_edgegroup<Index, DType>(
            H, feature_size, balan.keys, balan.balan_key.d_array.get(),
            balan.balan_row.d_array.get(), D1.d_array.get(), D2.d_array.get());
    }
    H.sp_data.download();
    bool pass = util::check_result_sddmm(
        H.nnz, H.sp_data.h_array.get(), out_ref.h_array.get());
    if (pass)
    {
        printf("check passed!\n");
    }
    return pass;
}

template <class Index, class DType, sddmm_kernel_met km, balan_met bm>
void SDDMM_test(std::fstream &fs, const int iter, int feature_size,
                SpMatCsrDescr_t<Index, DType> &H,
                gnn_balancer<Index, DType, bm> &balan,
                util::RamArray<DType> &in_feature1,
                util::RamArray<DType> &in_feature2)
{
    H.sp_data.fill_default_one();
    H.sp_data.upload();
    util::gpuTimer atimer;
    std::string method = "";
    atimer.start();
    if (km == sddmm_kernel_met::sddmm_edge_group)
    {
        for (int i = 0; i < iter; i++)
        {
            csrsddmm_edgegroup<Index, DType>(
                H, feature_size, balan.keys, balan.balan_key.d_array.get(),
                balan.balan_row.d_array.get(), in_feature1.d_array.get(), in_feature2.d_array.get());
        }
        method += "edge group";
    }
    atimer.end();
    float time = atimer.elapsed() / iter;
    std::cout << "The time of " << method << " sddmm " << time << std::endl;
    // fs << time << "," << 4 * feature_size * H.nnz * 1.0 / time / 1e6 << ",";
    fs << time << ",";
}

template <class Index, class DType, sddmm_kernel_met km>
float SDDMM_test(std::fstream &fs, const int iter, int feature_size,
                 SpMatCsrDescr_t<Index, DType> &H,
                 util::RamArray<DType> &in_feature1,
                 util::RamArray<DType> &in_feature2)
{
    H.sp_data.fill_default_one();
    H.sp_data.upload();
    util::gpuTimer atimer;
    std::string method = "";
    float compute_time = 0;
    atimer.start();
    if (km == sddmm_kernel_met::sddmm_cusparse)
    {
        compute_time += csrsddmm_cusparse_test<Index, DType>(
            iter, H, feature_size, in_feature1.d_array.get(), in_feature2.d_array.get());
        method += "cusparse";
    }
    else if (km == sddmm_kernel_met::sddmm_csrscale)
    {
        for (int i = 0; i < iter; ++i)
        {
            csrsddmm_scale<Index, DType>(
                H, feature_size, in_feature1.d_array.get(), in_feature2.d_array.get());
        }
        method += "csrscale";
    }
    atimer.end();
    float report_time = (km == sddmm_kernel_met::sddmm_cusparse)
                            ? (compute_time / iter)
                            : (atimer.elapsed() / iter);

    std::cout << "The time of " << method << " sddmm " << report_time
              << std::endl;
    // fs << report_time << "," << 4 * feature_size * H.nnz * 1.0 / report_time /
    // 1e6
    //    << ",";
    fs << report_time << ",";
    return report_time;
}

template <class Index, class DType, sddmm_kernel_met km>
bool SDDMM_check(int feature_size, SpMatCsrDescr_t<Index, DType> &H,
                 util::RamArray<DType> &in_feature1,
                 util::RamArray<DType> &in_feature2,
                 util::RamArray<DType> &out_ref)
{
    out_ref.reset();
    SDDMM_host<Index, DType>(feature_size, H, in_feature1, in_feature2, out_ref);
    if (km == sddmm_kernel_met::sddmm_cusparse)
    {
        csrsddmm_cusparse<Index, DType>(
            H.nrow, H.ncol, H.nnz, feature_size, H.sp_csrptr.d_array.get(),
            H.sp_csrind.d_array.get(), H.sp_data.d_array.get(),
            in_feature1.d_array.get(), in_feature2.d_array.get());
    }
    else if (km == sddmm_kernel_met::sddmm_csrscale)
    {
        csrsddmm_scale<Index, DType>(
            H, feature_size, in_feature1.d_array.get(), in_feature2.d_array.get());
    }
    H.sp_data.download();
    bool pass = util::check_result_sddmm(
        H.nnz, H.sp_data.h_array.get(), out_ref.h_array.get());
    if (pass)
    {
        printf("check passed!\n");
    }
    return pass;
}

#endif