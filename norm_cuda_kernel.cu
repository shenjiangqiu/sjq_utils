#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
template <typename scalar_t>
__global__ void norm_cuda_kernel(torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> weight,
                          torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> norm,
                          int n1, int n2,
                          int Wx,
                          int Wy,
                          int Nx,
                          int Ny )
{
  int startX = threadIdx.x * n1;
  int startY = threadIdx.y * n2;
  if(startX<Wx && startY<Wy){//valid thread
    scalar_t sum=0;
    scalar_t num=0;
    for(int i=0;i<n1;i++){
      for (int j=0;j<n2;j++){
        if(startX+i<Wx && startY+j<Wy){
          sum+=weight[startX+i][startY+j]*weight[startX+i][startY+j];
          num++;
        }
      }
    }
    norm[threadIdx.x][threadIdx.y]=sum/num;
  }
}

void norm_cuda(
    torch::Tensor weights,
    torch::Tensor out_norm,
    int n1,
    int n2)
{
  const auto WeightsSizeX = weights.size(0);
  const auto WeightsSizeY = weights.size(1);
  auto normSizeX=(WeightsSizeX+n1-1)/n1;
  auto normSizeY=(WeightsSizeY+n2-1)/n2;
  dim3 threadDim(8,8);
  dim3 blockDim((normSizeX+7/8),(normSizeY+7/8));
  
  

  AT_DISPATCH_FLOATING_TYPES(weights.type(), "norm_cuda", ([&] {
    norm_cuda_kernel<scalar_t><<<blockDim, threadDim>>>(
        weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        out_norm.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        n1,
        n2,
        WeightsSizeX,
        WeightsSizeY,
        normSizeX,
        normSizeY);
  }));
}