/**
 *   Copyright (c) 2026 Kioxia Corporation.
 *   All rights reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */
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
