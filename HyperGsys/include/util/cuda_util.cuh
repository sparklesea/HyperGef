#ifndef CUDA_UTIL
#define CUDA_UTIL

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define FULLMASK 0xffffffff
#define WARPSIZE 32
#define SHFL_DOWN_REDUCE(v)                                                    \
  v += __shfl_down_sync(FULLMASK, v, 16);                                      \
  v += __shfl_down_sync(FULLMASK, v, 8);                                       \
  v += __shfl_down_sync(FULLMASK, v, 4);                                       \
  v += __shfl_down_sync(FULLMASK, v, 2);                                       \
  v += __shfl_down_sync(FULLMASK, v, 1);

// end = start + how many seg parts
// itv(interval) = id in which idx of B's row
template <typename Index>
__device__ __forceinline__ void
__find_row_entry(Index id, Index *neighbor_key, Index *A_indices, Index start,
                 Index end, Index &B_row_idx, Index &itv) {
  Index lo = start, hi = end;
  // id is small, you could set the value already
  if (neighbor_key[lo] > id) {
    itv = id;
    B_row_idx = A_indices[lo];
    return;
  }
  while (lo < hi) {
    Index mid = (lo + hi) >> 1;
    if (__ldg(neighbor_key + mid) <= id) { // find the right(high)
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  // case lo = hi
  while (__ldg(neighbor_key + hi) == id) {
    ++hi;
  }
  B_row_idx = A_indices[hi];
  itv = id - neighbor_key[hi - 1];
}

__device__ __forceinline__ int findRow(const int *S_csrRowPtr, int eid,
                                       int start, int end)
{
  int low = start, high = end;
  if (low == high)
    return low;
  while (low < high)
  {
    int mid = (low + high) >> 1;
    if (S_csrRowPtr[mid] <= eid)
      low = mid + 1;
    else
      high = mid;
  }
  if (S_csrRowPtr[high] == eid)
    return high;
  else
    return high - 1;
}

template <typename ldType, typename data>
__device__ __forceinline__ void Load(ldType &tmp, data *array, int offset)
{
  tmp = *(reinterpret_cast<ldType *>(array + offset));
}

template <typename ldType, typename data>
__device__ __forceinline__ void Load(data *lhd, data *rhd, int offset)
{
  *(reinterpret_cast<ldType *>(lhd)) =
      *(reinterpret_cast<ldType *>(rhd + offset));
}

template <typename ldType, typename data>
__device__ __forceinline__ void Store(data *lhd, data *rhd, int offset)
{
  *(reinterpret_cast<ldType *>(lhd + offset)) =
      *(reinterpret_cast<ldType *>(rhd));
}

template <typename ldType, typename data>
__device__ __forceinline__ void Load4(ldType *tmp, data *array, int *offset,
                                      int offset2 = 0)
{
  Load(tmp[0], array, offset[0] + offset2);
  Load(tmp[1], array, offset[1] + offset2);
  Load(tmp[2], array, offset[2] + offset2);
  Load(tmp[3], array, offset[3] + offset2);
}

template <typename vecData, typename data>
__device__ __forceinline__ data vecDot2(vecData &lhd, vecData &rhd)
{
  return lhd.x * rhd.x + lhd.y * rhd.y;
}

template <typename vecData, typename data>
__device__ __forceinline__ data vecDot4(vecData &lhd, vecData &rhd)
{
  return lhd.x * rhd.x + lhd.y * rhd.y + lhd.z * rhd.z + lhd.w * rhd.w;
}

template <typename vecData, typename data>
__device__ __forceinline__ void vec4Dot4(data *cal, vecData *lhd,
                                         vecData *rhd)
{
  cal[0] += vecDot4<vecData, data>(lhd[0], rhd[0]);
  cal[1] += vecDot4<vecData, data>(lhd[1], rhd[1]);
  cal[2] += vecDot4<vecData, data>(lhd[2], rhd[2]);
  cal[3] += vecDot4<vecData, data>(lhd[3], rhd[3]);
}

template <typename vecData, typename data>
__device__ __forceinline__ void vec2Dot4(data *cal, vecData *lhd,
                                         vecData *rhd)
{
  cal[0] += vecDot2<vecData, data>(lhd[0], rhd[0]);
  cal[1] += vecDot2<vecData, data>(lhd[1], rhd[1]);
  cal[2] += vecDot2<vecData, data>(lhd[2], rhd[2]);
  cal[3] += vecDot2<vecData, data>(lhd[3], rhd[3]);
}

template <typename data>
__device__ __forceinline__ void Dot4(data *cal, data *lhd, data *rhd)
{
  cal[0] += lhd[0] * rhd[0];
  cal[1] += lhd[1] * rhd[1];
  cal[2] += lhd[2] * rhd[2];
  cal[3] += lhd[3] * rhd[3];
}

template <typename data>
__device__ __forceinline__ void selfMul4(data *lhd, data *rhd)
{
  lhd[0] *= rhd[0];
  lhd[1] *= rhd[1];
  lhd[2] *= rhd[2];
  lhd[3] *= rhd[3];
}

template <typename data>
__device__ __forceinline__ void selfMulConst4(data *lhd, data Const)
{
  lhd[0] *= Const;
  lhd[1] *= Const;
  lhd[2] *= Const;
  lhd[3] *= Const;
}

template <typename data>
__device__ __forceinline__ void selfAddConst4(data *lhd, data Const)
{
  lhd[0] += Const;
  lhd[1] += Const;
  lhd[2] += Const;
  lhd[3] += Const;
}

template <typename data>
__device__ __forceinline__ void AllReduce4(data *multi, int stride,
                                           int warpSize)
{
  for (; stride > 0; stride >>= 1)
  {
    multi[0] += __shfl_xor_sync(0xffffffff, multi[0], stride, warpSize);
    multi[1] += __shfl_xor_sync(0xffffffff, multi[1], stride, warpSize);
    multi[2] += __shfl_xor_sync(0xffffffff, multi[2], stride, warpSize);
    multi[3] += __shfl_xor_sync(0xffffffff, multi[3], stride, warpSize);
  }
}

#endif // CUDA_UTIL