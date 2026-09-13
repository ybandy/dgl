#include <graphbolt/async.h>
#include <torch/script.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <mutex>

#include "./utils.h"

#define NUM_BUFFERS (3)


struct qpair_context {
  struct spdk_nvme_ns *ns;
  struct spdk_nvme_qpair *qpair;
  uint64_t last_index;
  uint64_t last_data_index;
  uint64_t num_submitted;
  uint64_t num_completed;
  void *node_ids; // void* to avoid type template
  std::byte *data;
};

struct io_context {
  struct qpair_context *qctx;
  uint32_t drive_id;
  uint32_t num_drives;
  uint64_t index;
  uint64_t end;
  uint64_t data_index;
  uint64_t feature_size;
  uint64_t feature_stride;
  void *buf;
};


struct qpair_context_overlapped {
  struct spdk_nvme_ns *ns;
  struct spdk_nvme_qpair *qpair;
  uint64_t last_index[NUM_BUFFERS];
  uint64_t num_submitted[NUM_BUFFERS];
  uint64_t num_completed[NUM_BUFFERS];
  void *node_ids[NUM_BUFFERS]; // void* to avoid type template
  std::byte *data[NUM_BUFFERS];
  std::atomic<bool> *ready[NUM_BUFFERS];
  uint64_t end[NUM_BUFFERS];
};

struct io_context_overlapped {
  struct qpair_context_overlapped *qctx;
  uint32_t drive_id;
  uint32_t num_drives;
  uint64_t index;
  uint64_t end;
  uint64_t data_index;
  int32_t toggle;
  uint32_t feature_size;
  uint64_t feature_stride;
  void *buf;
};



namespace graphbolt {
namespace storage {

class SPDK : public torch::CustomClassHolder {

 public:

  enum IndexSelectMode
  {
    LAUNCH_THREADS = 0,
    PERSIST_THREADS_GUARDED = 1,
    PERSIST_THREADS = 2,
    PERSIST_THREADS_OVERLAPPED = 3
  };

  SPDK() = default;

  SPDK(
      torch::ScalarType dtype,
      const std::vector<int64_t>& shape, torch::optional<int64_t> num_threads,
      torch::optional<int64_t> num_contexts, torch::optional<int64_t> index_select_mode);

  static c10::intrusive_ptr<SPDK> Create(
      torch::ScalarType dtype,
      const std::vector<int64_t>& shape, torch::optional<int64_t> num_threads,
      torch::optional<int64_t> num_contexts, torch::optional<int64_t> index_select_mode);

  ~SPDK();

  c10::intrusive_ptr<Future<torch::Tensor>> IndexSelect(torch::Tensor index, int64_t minibatch_idx);
  
 private:

  torch::Tensor IndexSelectImpl(torch::Tensor index, int64_t minibatch_idx);

  template <typename index_t>
  inline void IndexSelectImplLaunchThreads(torch::Tensor index, int64_t minibatch_idx, torch::Tensor& result);

  inline void IndexSelectImplRunPersistentThreads(torch::Tensor index, int64_t minibatch_idx, torch::Tensor& result);
  inline void IndexSelectImplRunPersistentThreadsOverlapped(torch::Tensor index, int64_t minibatch_idx, torch::Tensor& result);

  template <typename index_t>
  void IndexSelectWorker(int64_t thread_id, int64_t qid, int64_t drive_id,
                         int64_t begin, int64_t end,
                         index_t *index_data, std::byte *result_buffer);

  void InitContexts();
  void DestroyContexts();

  void IndexSelectWorkerPersistent(int64_t thread_id);
  void IndexSelectWorkerPersistentOverlapped(int64_t thread_id);

  void LaunchPersistentThreads();
  void JoinPersistentThreads();

  struct qpair_context *qctx = nullptr;
  struct io_context **ioctx = nullptr;

  struct qpair_context_overlapped *qctx_ovl = nullptr;

  std::mutex mutex_;

  std::promise<void> promise_;
  std::future<void> future_;
  std::exception_ptr eptr_;
  int num_persistent_threads_launched_;
  std::atomic<int> num_persistent_threads_finished_;
  bool persistent_threads_running_;
  
  std::atomic<uint64_t> toggle_;
  std::atomic<bool> ready_[NUM_BUFFERS];
  std::byte *result_buffer_[NUM_BUFFERS];
  torch::Tensor index_[NUM_BUFFERS];
  std::promise<void>* persistent_threads_promise_[NUM_BUFFERS];
  std::atomic<int> num_work_finished_[NUM_BUFFERS];

  std::mutex mtx_print_;

  const std::vector<int64_t>
      feature_dim_;                // Shape of features, e.g. {N,M,K,L}.
  const torch::ScalarType dtype_;  // Feature data type.
  const int64_t feature_size_;     // Number of bytes of feature size.
  const int64_t num_contexts_;
  const int64_t num_qpairs_;
  const IndexSelectMode index_select_mode_;

  utils::TimeStamp* timestamp;
};

}  // namespace storage
}  // namespace graphbolt
