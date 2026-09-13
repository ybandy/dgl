#ifndef GRAPHBOLT_PIN_MEMORY_H_
#define GRAPHBOLT_PIN_MEMORY_H_

#include <graphbolt/async.h>

namespace graphbolt {
namespace ops {

std::vector<torch::Tensor> PinMemory(const std::vector<torch::Tensor>& tensors);

c10::intrusive_ptr<Future<std::vector<torch::Tensor>>> PinMemoryAsync(const std::vector<torch::Tensor>& tensors);

}  // namespace ops
}  // namespace graphbolt

#endif  // GRAPHBOLT_PIN_MEMORY_H_
