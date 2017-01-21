/*
    This file is part of ParTI!.

    ParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    ParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with ParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <ParTI.h>
#include "sptensor.h"
#include "../cudawrap.h"

__global__ static void spt_MTTKRPKernel(
    const size_t mode,
    const size_t nmodes,
    const size_t nnz,
    const size_t R,
    const size_t stride,
    const size_t * Xndims,
    size_t ** const Xinds,
    const sptScalar * Xvals,
    const size_t * dev_mats_order,
    sptScalar ** dev_mats,
    sptScalar * dev_scratch
) {
    const size_t tidx = threadIdx.x;
    const size_t x = blockIdx.x * blockDim.x + tidx;

    size_t const nmats = nmodes - 1;
    size_t const * const mode_ind = Xinds[mode];
    sptScalar * const mvals = (sptScalar*) dev_mats[nmodes];

    if(x < nnz) {
        size_t times_mat_index = dev_mats_order[0];
        sptScalar * times_mat = dev_mats[times_mat_index];
        size_t * times_inds = Xinds[times_mat_index];
        size_t tmp_i = times_inds[x];
        sptScalar const entry = Xvals[x];
        for(size_t r=0; r<R; ++r) {
            dev_scratch[x * stride + r] = entry * times_mat[tmp_i * stride + r];
        }

        for(size_t i=1; i<nmats; ++i) {
            times_mat_index = dev_mats_order[i];
            times_mat = dev_mats[times_mat_index];
            times_inds = Xinds[times_mat_index];
            tmp_i = times_inds[x];
            for(size_t r=0; r<R; ++r) {
                dev_scratch[x * stride + r] *= times_mat[tmp_i * stride + r];
            }
        }

    }

    __syncthreads();

    if(x < nnz) {
        size_t const mode_i = mode_ind[x];
        for(size_t r=0; r<R; ++r) {
            atomicAdd(&(mvals[mode_i * stride + r]), dev_scratch[x * stride + r]);
        }
    }
    __syncthreads();
}




/**
 * CUDA parallelized Matriced sparse tensor times a sequence of dense matrix Khatri-Rao products (MTTKRP) on a specified mode
 * @param[out] mats[nmodes]    the result of MTTKRP, a dense matrix, with size
 * ndims[mode] * R
 * @param[in]  X    the sparse tensor input X
 * @param[in]  mats    (N+1) dense matrices, with mats[nmodes] as temporary
 * @param[in]  mats_order    the order of the Khatri-Rao products
 * @param[in]  mode   the mode on which the MTTKRP is performed
 * @param[in]  scratch an temporary array to store intermediate results, space assigned before this function
 *
 * This function uses support arbitrary-order sparse tensors with Khatri-Rao
 * products of dense factor matrices, the output is the updated dense matrix for the "mode".
 * In this version, atomic function to lock the global reduction and a large
 * scratch is used to maximize parallelism. (To be optimized)
 */
int sptPresplittedMTTKRP(
    sptSparseTensor const splits[],
    size_t const nsplits,
    size_t const batch_size,
    sptMatrix *mats[],
    size_t const mats_order[],
    size_t const mode
) {
    if(nsplits == 0) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA SpTns SpltMTTKRP", "nsplits == 0");
    }

    size_t nmodes = splits[0].nmodes;
    const size_t *ndims = splits[0].ndims;
    size_t R = mats[mode]->ncols;
    size_t stride = mats[mode]->stride;
    size_t nmats = nmodes - 1;
    // TODO: chek nmats
    int result;

    /* Check the mats. */
    for(size_t i = 0; i < nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA SpTns SpltMTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
    }

    sptMatrix product;
    result = sptNewMatrix(&product, mats[mode]->nrows, mats[mode]->ncols);
    spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);
    memset(product.values, 0, product.nrows * product.stride * sizeof (sptScalar));

    size_t *dev_Xndims;
    result = sptCudaDuplicateMemory(&dev_Xndims, ndims, nmodes * sizeof *dev_Xndims, cudaMemcpyHostToDevice);
    spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

    size_t *dev_mats_order;
    result = sptCudaDuplicateMemory(&dev_mats_order, mats_order, nmodes * sizeof *dev_mats_order, cudaMemcpyHostToDevice);

    sptScalar ***dev_mats = new sptScalar **[batch_size];
    for(size_t i = 0; i < batch_size; ++i) {
        // TODO: dev_mats, be aware of the last one
    }

    size_t batch_count = (nsplits-1)/batch_size + 1;
    size_t ***dev_Xinds = new size_t **[batch_size];
    sptScalar **dev_Xvals = new sptScalar *[batch_size];
    sptScalar **dev_scratch = new sptScalar *[batch_size];

    for(size_t batch_idx = 0; batch_idx < batch_count; ++batch_idx) {
        size_t kernel_count = batch_idx == batch_count-1 ? nsplits - batch_idx*batch_size : batch_size;
        for(size_t kernel_idx = 0; kernel_idx < kernel_count; ++kernel_idx) {
            size_t split_idx = batch_idx*batch_size + kernel_count;

            // TODO: dev_Xinds

            result = sptCudaDuplicateMemory(&dev_Xvals[kernel_idx], splits[split_idx].values.data, splits[split_idx].nnz * sizeof (sptScalar), cudaMemcpyHostToDevice);
            spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

            result = cudaMalloc((void **) &dev_scratch[kernel_idx], splits[split_idx].nnz * stride * sizeof (sptScalar));
            spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
        }

        for(size_t kernel_idx = 0; kernel_idx < kernel_count; ++kernel_idx) {
            size_t split_idx = batch_idx*batch_size + kernel_count;
            size_t nnz = splits[split_idx].nnz;

            size_t nthreads = 128;
            size_t nblocks = (nnz + nthreads -1) / nthreads;

            spt_MTTKRPKernel<<<nblocks, nthreads>>>(
                mode,
                nmodes,
                nnz,
                R,
                stride,
                dev_Xndims,
                dev_Xinds[kernel_idx],
                dev_Xvals[kernel_idx],
                dev_mats_order,
                dev_mats[kernel_idx],
                dev_scratch[kernel_idx]
            );
        }

        result = cudaDeviceSynchronize();
        spt_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");

        for(size_t kernel_idx = 0; kernel_idx < kernel_count; ++kernel_idx) {

            // TODO: copy dev_mats[nmodes] back

            for(size_t i = 0; i < product.nrows * product.stride; ++i) {
                product.values[i] += mats[nmodes]->values[i];
            }
        }

        for(size_t kernel_idx = 0; kernel_idx < kernel_count; ++kernel_idx) {
            result = cudaFree(dev_scratch[kernel_idx]);
            result = cudaFree(dev_Xvals[kernel_idx]);

            // TODO: dev_Xinds
        }
    }

    delete[] dev_scratch;
    delete[] dev_Xvals;
    delete[] dev_Xinds;

    // TODO: dev_mats

    delete[] dev_mats;
    result = cudaFree(dev_mats_order);
    spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);
    result = cudaFree(dev_Xndims);
    spt_CheckError(result, "CUDA SpTns SpltMTTKRP", NULL);

    memcpy(mats[splits[0].nmodes]->values, product.values, product.nrows * product.stride);
    sptFreeMatrix(&product);

    return 0;
}
