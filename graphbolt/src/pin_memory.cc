#include "./pin_memory.h"


namespace graphbolt {
namespace ops {


std::vector<torch::Tensor> PinMemory(const std::vector<torch::Tensor>& tensors) {
  std::vector<torch::Tensor> results;
  results.reserve(tensors.size());
  for (auto tensor : tensors) {
    results.emplace_back(tensor.pin_memory());
  }
  return results;
}

c10::intrusive_ptr<Future<std::vector<torch::Tensor>>> PinMemoryAsync(const std::vector<torch::Tensor>& tensors) {
  return async([=] { return PinMemory(tensors); });
}


}  // namespace ops
}  // namespace graphbolt
