/**
 *   Copyright (c) 2024, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
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
 *
 * @file utils.cc
 * @brief Graphbolt utils implementations.
 */
#include "./utils.h"

#include <optional>
#include <fstream>

namespace graphbolt {
namespace utils {

namespace {
std::optional<int64_t> worker_id;
}

std::optional<int64_t> GetWorkerId() { return worker_id; }

void SetWorkerId(int64_t worker_id_value) { worker_id = worker_id_value; }


TimeStamp::TimeStamp(std::string label)
    : label_(label) {
  start_times_.resize(max_num_timestamps_, std::chrono::nanoseconds::zero());
  end_times_.resize(max_num_timestamps_, std::chrono::nanoseconds::zero());
  //std::optional<int64_t> worker_id = GetWorkerId();
  //std::string filename = "timestamp_" + label_;
  //if(worker_id.has_value()) filename += std::to_string(worker_id.value());
  //filename += std::to_string(getpid());
  //filename += ".txt";
  //std::ofstream file(filename);
  //file.close();
}

TimeStamp::~TimeStamp() {
  //std::optional<int64_t> worker_id = GetWorkerId();
  std::string filename = "timestamp_" + label_;
  //if(worker_id.has_value()) filename += std::to_string(worker_id.value());
  filename += ".txt";
  //std::cerr << "time stamp destructor called " + filename << std::endl;
  std::ofstream file(filename);
  for(int64_t i = 0; i < max_num_timestamps_; i++) {
    if(start_times_[i].count() == 0) break;
    dump(file, i, start_times_[i], "start");
    dump(file, i,   end_times_[i], "end");
  }
  file.close();
  //std::cerr << "time stamp destructor file close " + filename << std::endl;
}

void TimeStamp::set_num_layers(int64_t num_layers) { num_layers_ = num_layers; }

void TimeStamp::record_start(int64_t minibatch_idx, int64_t layer_idx) {
    record_time(minibatch_idx, layer_idx, start_times_);
}

void TimeStamp::record_end(int64_t minibatch_idx, int64_t layer_idx) {
    record_time(minibatch_idx, layer_idx, end_times_);
}

void TimeStamp::record_time(int64_t minibatch_idx, int64_t layer_idx,
                            std::vector<std::chrono::nanoseconds> &times) {
  int64_t idx = num_layers_ <= 0 ? minibatch_idx : minibatch_idx * num_layers_ + layer_idx;
  if(0 <= idx && idx < max_num_timestamps_ && times[idx] == std::chrono::nanoseconds::zero()) {
    times[idx] = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
  }
}

void TimeStamp::dump(std::ostream &ost, int64_t idx, std::chrono::nanoseconds time, std::string when) {
  if(num_layers_ <= 0) {
    ost << label_ << "(" << idx << ") " << when << ": " << time.count() << std::endl;
  } else {
    int64_t minibatch_idx = idx / num_layers_;
    int64_t layer_idx = idx % num_layers_;
    ost << label_ << "(" << minibatch_idx << "," << layer_idx << ") " << when << ": " << time.count() << std::endl;
  }
}

}  // namespace utils
}  // namespace graphbolt
