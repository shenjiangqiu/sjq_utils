#include <torch/extension.h>

// CUDA interface
void norm_cuda(
    torch::Tensor weights,
    torch::Tensor norms,
    int n1,
    int n2);
// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

void norm(
    torch::Tensor weights,
    torch::Tensor norms,
    int n1,
    int n2)
{
  CHECK_INPUT(weights);
  CHECK_INPUT(norms);
  norm_cuda(weights, norms, n1, n2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("norm", &norm, "in-place norm");
}