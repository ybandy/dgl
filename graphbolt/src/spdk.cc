#include "./spdk.h"

#include <graphbolt/async.h>
#include <torch/torch.h>

#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <thread>

#include "./utils.h"
#include "./spdk_util.h"
#include "spdk/likely.h"


using spdk_util::drive;

#define BLOCK_SIZE (512UL)

static inline void set_affinity(int64_t core_id)
{
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  CPU_SET(core_id, &cpu_set);
  sched_setaffinity(0, sizeof(cpu_set_t), &cpu_set);
}


namespace graphbolt {
namespace storage {


SPDK::SPDK(
    torch::ScalarType dtype,
    const std::vector<int64_t> &shape, torch::optional<int64_t> num_threads,
    torch::optional<int64_t> num_contexts, torch::optional<int64_t> index_select_mode)
    : feature_dim_(shape),
      dtype_(dtype),
      feature_size_(std::accumulate(
          shape.begin() + 1, shape.end(), c10::elementSize(dtype),
          std::multiplies<int64_t>())),
      num_qpairs_(num_threads.value_or(1)),
      num_contexts_(num_contexts.value_or(1)),
      index_select_mode_(static_cast<IndexSelectMode>(index_select_mode.value_or(0))) {
  if(feature_size_ > BLOCK_SIZE) { //featureStride) {
    throw std::runtime_error(
        "feature_size " + std::to_string(feature_size_)
        + " must be no greater than " + std::to_string(BLOCK_SIZE));
  }
  unsigned int num_lcores = std::thread::hardware_concurrency();
  char lcore_map[128];
  sprintf(lcore_map, "0-%d", num_lcores - 1);
  printf("lcore_map: %s\n", lcore_map);
  spdk_util::init(num_qpairs_, num_contexts_, lcore_map);
  spdk_unaffinitize_thread();
  std::cout << "size of struct qpair_context: " << sizeof(struct qpair_context) << std::endl;
  std::cout << "size of struct io_context: " << sizeof(struct io_context) << std::endl;
  assert(sizeof(struct qpair_context) == 64);
  assert(sizeof(struct io_context) == 64);
  InitContexts();
  LaunchPersistentThreads();
  timestamp = new utils::TimeStamp("IndexSelect_SPDK");
}

c10::intrusive_ptr<SPDK> SPDK::Create(
    torch::ScalarType dtype,
    const std::vector<int64_t> &shape, torch::optional<int64_t> num_threads,
    torch::optional<int64_t> num_contexts, torch::optional<int64_t> index_select_mode) {
  return c10::make_intrusive<SPDK>(dtype, shape, num_threads, num_contexts, index_select_mode);
}

SPDK::~SPDK() {
  DestroyContexts();
  JoinPersistentThreads();
 	spdk_util::fini();
  delete timestamp;
}

c10::intrusive_ptr<Future<torch::Tensor>> SPDK::IndexSelect(
    torch::Tensor index, int64_t minibatch_idx) {
  return async([=, this] { return IndexSelectImpl(index, minibatch_idx); });
}

torch::Tensor SPDK::IndexSelectImpl(torch::Tensor index, int64_t minibatch_idx) {

  timestamp->record_start(minibatch_idx);

  std::vector<int64_t> shape(index.sizes().begin(), index.sizes().end());
  shape.insert(shape.end(), feature_dim_.begin() + 1, feature_dim_.end());
  auto result = torch::empty(
      shape, index.options()
                  .dtype(dtype_)
                  .layout(torch::kStrided)
                  //.pinned_memory(utils::is_pinned(index))
                  .pinned_memory(true) // force pinned memory for fetched features
                  .requires_grad(false));

  if(index_select_mode_ == LAUNCH_THREADS) {
    AT_DISPATCH_INDEX_TYPES(
        index.scalar_type(), "IndexSelectImpl", ([&] {
          IndexSelectImplLaunchThreads<index_t>(index, minibatch_idx, result);
        }));
  } else if(index_select_mode_ == PERSIST_THREADS_GUARDED) {
    std::lock_guard<std::mutex> guard(mutex_);
    IndexSelectImplRunPersistentThreads(index, minibatch_idx, result);
  } else if(index_select_mode_ == PERSIST_THREADS) {
    IndexSelectImplRunPersistentThreads(index, minibatch_idx, result);
  } else if(index_select_mode_ == PERSIST_THREADS_OVERLAPPED) {
    IndexSelectImplRunPersistentThreadsOverlapped(index, minibatch_idx, result);
  }

  timestamp->record_end(minibatch_idx);
  return result;
}

template <typename index_t>
void SPDK::IndexSelectImplLaunchThreads(torch::Tensor index, int64_t minibatch_idx, torch::Tensor &result)
{
  auto *result_buffer = reinterpret_cast<std::byte *>(result.data_ptr());
  const int64_t num_drives = spdk_util::num_drives();
  const int64_t num_threads = num_qpairs_ * num_drives;
  auto index_data = index.data_ptr<index_t>();
  std::lock_guard<std::mutex> guard(mutex_); // make sure no two parallel_for's at the same time
  graphbolt::parallel_for_each(0, num_threads, 1, [&](int thread_id) {
    int64_t qid = thread_id / num_drives;
    int64_t drive_id = thread_id % num_drives;
    int64_t begin = index.numel() * qid / num_qpairs_;
    int64_t end = index.numel() * (qid + 1) / num_qpairs_;
    IndexSelectWorker<index_t>(thread_id, qid, drive_id, begin, end, index_data, result_buffer);
  });
}

void SPDK::IndexSelectImplRunPersistentThreads(torch::Tensor index, int64_t minibatch_idx, torch::Tensor &result)
{
  const int i = toggle_.fetch_add(1, std::memory_order_seq_cst) % NUM_BUFFERS;
  index_[i] = index;
  result_buffer_[i] = reinterpret_cast<std::byte *>(result.data_ptr());
  persistent_threads_promise_[i] = new std::promise<void>();
  std::future<void> f = persistent_threads_promise_[i]->get_future();
  num_work_finished_[i] = 0;
  //ready_[i] = true; // let the persistent threads start processing the minibatch
  ready_[i].store(true, std::memory_order_release); // let the persistent threads start processing the minibatch

  f.get(); // wait for the threads to collectively finish the work
  delete persistent_threads_promise_[i];
}

void SPDK::IndexSelectImplRunPersistentThreadsOverlapped(torch::Tensor index, int64_t minibatch_idx, torch::Tensor &result)
{
  const int i = toggle_.fetch_add(1, std::memory_order_seq_cst) % NUM_BUFFERS;
  index_[i] = index;
  result_buffer_[i] = reinterpret_cast<std::byte *>(result.data_ptr());
  persistent_threads_promise_[i] = new std::promise<void>();
  std::future<void> f = persistent_threads_promise_[i]->get_future();
  num_work_finished_[i] = 0;
  const int64_t num_drives = spdk_util::num_drives();
  const int64_t num_threads = num_qpairs_ * num_drives;
  for(int j = 0; j < num_threads; j++)
  {
    int64_t qid = j / num_drives;
    //qctx_ovl[j].num_submitted[i] = 0;
    //qctx_ovl[j].num_completed[i] = 0;
    qctx_ovl[j].node_ids[i] = index.data_ptr();
    qctx_ovl[j].data[i] = result_buffer_[i];
    qctx_ovl[j].last_index[i] = index.numel() * qid / num_qpairs_;
    qctx_ovl[j].end[i] = index.numel() * (qid + 1) / num_qpairs_;
  }
  ready_[i] = true; // let the persistent threads start processing the minibatch

  f.get(); // wait for the threads to collectively finish the work
  delete persistent_threads_promise_[i];
}


template <typename index_t>
static void submit_io(struct io_context *ctx);

template <typename index_t>
static void read_complete(void *arg, const struct spdk_nvme_cpl *completion)
{
	struct io_context *ctx = (struct io_context *)arg;
  struct qpair_context *qctx = ctx->qctx;
  uint64_t index = qctx->last_index;

  if(spdk_unlikely(spdk_nvme_cpl_is_error(completion))) {
		fprintf(stderr, "I/O error status: %s\n", spdk_nvme_cpl_get_status_string(&completion->status));
		fprintf(stderr, "Read I/O failed, aborting run\n");
		exit(1);
	}

  memcpy((uint8_t *)qctx->data + ctx->index * ctx->feature_size, ctx->buf, ctx->feature_size);

  qctx->num_completed++;
  index_t *node_ids = (index_t *)qctx->node_ids;
  for(; index < ctx->end; index++) {
    if(ctx->drive_id == node_ids[index] % ctx->num_drives) break;
  }
  if(spdk_likely(index < ctx->end)) {
    ctx->index = index;
    //ctx->data_index = qctx->last_data_index + 1;
    submit_io<index_t>(ctx);
  } else {
    qctx->last_index = ctx->end;
  }
}

template <typename index_t>
static void submit_io(struct io_context *ctx)
{
  struct qpair_context *qctx = ctx->qctx;
  const uint32_t num_drives = ctx->num_drives;
  index_t node_id = ((index_t *)qctx->node_ids)[ctx->index];
  assert(ctx->drive_id == node_id % num_drives);
  uint64_t lba = node_id / num_drives;
  qctx->num_submitted++;
  qctx->last_index = ctx->index + 1;
  qctx->last_data_index = ctx->data_index;
  int rc = spdk_nvme_ns_cmd_read(qctx->ns, qctx->qpair,
                                 ctx->buf,
                                 lba, 1, read_complete<index_t>, ctx, 0);
  if(spdk_unlikely(rc != 0))
  {
    fprintf(stderr, "starting read I/O failed\n");
    exit(1);
  }
}

//-------------------------------------------------------------------------
template <typename index_t>
static void submit_io_overlapped(struct io_context_overlapped *ctx);

template <typename index_t>
static void read_complete_overlapped(void *arg, const struct spdk_nvme_cpl *completion)
{
	struct io_context_overlapped *ctx = (struct io_context_overlapped *)arg;
  struct qpair_context_overlapped *qctx = ctx->qctx;
  const int32_t toggle = ctx->toggle;
  assert(0 <= toggle && toggle < NUM_BUFFERS);
  uint64_t index = qctx->last_index[toggle];// + 1;

  if(spdk_unlikely(spdk_nvme_cpl_is_error(completion))) {
		fprintf(stderr, "I/O error status: %s\n", spdk_nvme_cpl_get_status_string(&completion->status));
		fprintf(stderr, "Read I/O failed, aborting run\n");
		exit(1);
	}

  memcpy((uint8_t *)qctx->data[toggle] + ctx->index * ctx->feature_size, ctx->buf, ctx->feature_size);

  qctx->num_completed[toggle]++;
  index_t *node_ids = (index_t *)qctx->node_ids[toggle];
  for(; index < ctx->end; index++) {
    if(ctx->drive_id == node_ids[index] % ctx->num_drives) break;
  }
  if(spdk_likely(index < ctx->end)) {
    ctx->index = index;
    //ctx->data_index = qctx->last_data_index + 1;
    submit_io_overlapped<index_t>(ctx);
  } else {
    qctx->last_index[toggle] = ctx->end;
    const int32_t next = (toggle + 1) % NUM_BUFFERS;
    if(qctx->ready[next]->load(std::memory_order_acquire))
    {
      index = qctx->last_index[next];
      node_ids = (index_t *)qctx->node_ids[next];
      for(; index < qctx->end[next]; index++) {
        if(ctx->drive_id == node_ids[index] % ctx->num_drives) break;
      }
      if(spdk_likely(index < qctx->end[next]))
      {
        ctx->index = index;
        ctx->end = qctx->end[next];
        //ctx->data_index = i;
        ctx->toggle = next;
        submit_io_overlapped<index_t>(ctx);
      }
      else { ctx->toggle = -1; }
    }
    else { ctx->toggle = -1; }
  }
}

template <typename index_t>
static void submit_io_overlapped(struct io_context_overlapped *ctx)
{
  struct qpair_context_overlapped *qctx = ctx->qctx;
  const uint32_t num_drives = ctx->num_drives;
  const int32_t toggle = ctx->toggle;
  assert(0 <= toggle && toggle < NUM_BUFFERS);
  index_t node_id = ((index_t *)qctx->node_ids[toggle])[ctx->index];
  assert(ctx->drive_id == node_id % num_drives);
  uint64_t lba = node_id / num_drives;
  qctx->num_submitted[toggle]++;
  qctx->last_index[toggle] = ctx->index + 1;
  //qctx->last_data_index[toggle] = ctx->data_index;
  int rc = spdk_nvme_ns_cmd_read(qctx->ns, qctx->qpair,
                                 ctx->buf,
                                 lba, 1, read_complete_overlapped<index_t>, ctx, 0);
  if(spdk_unlikely(rc != 0))
  {
    fprintf(stderr, "starting read I/O failed\n");
    exit(1);
  }
}


template <typename index_t>
void SPDK::IndexSelectWorker(int64_t thread_id, int64_t qid, int64_t drive_id,
                             int64_t begin, int64_t end,
                             index_t *index_data, std::byte *result_buffer) {
  set_affinity(thread_id);

  const int num_drives = spdk_util::num_drives();

  struct qpair_context *qc = qctx + thread_id;
  qc->last_index = 0;
  qc->num_submitted = 0;
  qc->num_completed = 0;
  qc->node_ids = (void *)index_data;
  qc->data = result_buffer;

  uint64_t index = begin;
  for(uint32_t i = 0; i < num_contexts_; i++)
  {
      for(; index < end; index++) {
          if(drive_id == index_data[index] % num_drives) break;
      }
      if(index >= end) break;

      struct io_context *ctx = ioctx[thread_id] + i;
      ctx->index = index;
      ctx->end = end;
      ctx->data_index = i;
      submit_io<index_t>(ctx);
      index++;
  }

  while(spdk_likely(qc->last_index < end || qc->num_completed < qc->num_submitted))
  {
    spdk_nvme_qpair_process_completions(qc->qpair, 0);
  }

  assert(qc->num_submitted == qc->num_completed);
}

void SPDK::InitContexts()
{
  const int num_drives = spdk_util::num_drives();
  const int num_threads = num_qpairs_ * num_drives;
  if(index_select_mode_ == LAUNCH_THREADS)
  {
    qctx = new struct qpair_context[num_threads];
    ioctx = new struct io_context*[num_threads];
    for(int qid = 0; qid < num_qpairs_; qid++)
    {
      for(int drive_id = 0; drive_id < num_drives; drive_id++)
      {
        int thread_id = qid * num_drives + drive_id;
        struct drive *drive = spdk_util::get_drive(drive_id);
        const int32_t numa_id = spdk_nvme_ctrlr_get_numa_id(drive->ctrlr);

        struct qpair_context *qc = qctx + thread_id;
        qc->ns = drive->ns;
        qc->qpair = drive->qpairs[qid];

        ioctx[thread_id] = new struct io_context[num_contexts_];
        for(uint32_t i = 0; i < num_contexts_; i++)
        {
            struct io_context *ic = ioctx[thread_id] + i;
            ic->qctx = qc;
            ic->drive_id = drive_id;
            ic->num_drives = num_drives;
            ic->feature_size = feature_size_;
            //ic->feature_stride = featureStride;
            ic->buf = spdk_zmalloc(BLOCK_SIZE, BLOCK_SIZE, NULL, numa_id, SPDK_MALLOC_DMA);
        }
      }
    }
  }
  else if(index_select_mode_ == PERSIST_THREADS_OVERLAPPED)
  {
    qctx_ovl = new struct qpair_context_overlapped[num_threads];
  }
}

void SPDK::DestroyContexts()
{
  const int num_drives = spdk_util::num_drives();
  const int num_threads = num_qpairs_ * num_drives;
  if(ioctx)
  {
    for(int i = 0; i < num_threads; i++)
    {
      for(int j = 0; j < num_contexts_; j++) spdk_free(ioctx[i][j].buf);
      delete [] ioctx[i];
    }
    delete [] ioctx;
    ioctx = nullptr;
  }

  if(qctx)
  {
    delete [] qctx;
    qctx = nullptr;
  }

  if(qctx_ovl)
  {
    delete [] qctx_ovl;
    qctx_ovl = nullptr;
  }
}


void SPDK::IndexSelectWorkerPersistent(int64_t thread_id)
{
  set_affinity(thread_id + 64);

  const int64_t num_drives = spdk_util::num_drives();
  const int64_t qid = thread_id / num_drives;
  const int64_t drive_id = thread_id % num_drives;

  struct drive *drive = spdk_util::get_drive(drive_id);
  const int32_t numa_id = spdk_nvme_ctrlr_get_numa_id(drive->ctrlr);

  struct qpair_context qctx;
  qctx.ns = drive->ns;
  qctx.qpair = drive->qpairs[qid];

  struct io_context *ioctx = new struct io_context[num_contexts_];
  for(uint32_t i = 0; i < num_contexts_; i++)
  {
      struct io_context *ic = ioctx + i;
      ic->qctx = &qctx;
      ic->drive_id = drive_id;
      ic->num_drives = num_drives;
      ic->feature_size = feature_size_;
      //ic->feature_stride = featureStride;
      ic->buf = spdk_zmalloc(BLOCK_SIZE, BLOCK_SIZE, NULL, numa_id, SPDK_MALLOC_DMA);
  }

  int toggle = 0;
  while(persistent_threads_running_)
  {
    while(persistent_threads_running_ && !ready_[toggle].load(std::memory_order_acquire)) ; // loop
    if(!persistent_threads_running_) break;
  
    int64_t num_ids = index_[toggle].numel();
    int64_t begin = num_ids * qid / num_qpairs_;
    int64_t end = num_ids * (qid + 1) / num_qpairs_;

    qctx.last_index = 0;
    qctx.num_submitted = 0;
    qctx.num_completed = 0;
    qctx.node_ids = index_[toggle].data_ptr();
    qctx.data = result_buffer_[toggle];

    AT_DISPATCH_INDEX_TYPES(
        index_[toggle].scalar_type(), "IndexSelectImpl", ([&] {
          uint64_t index = begin;
          for(uint32_t i = 0; i < num_contexts_; i++)
          {
              for(; index < end; index++) {
                  if(drive_id == reinterpret_cast<index_t *>(qctx.node_ids)[index] % num_drives) break;
              }
              if(index >= end) break;
              struct io_context *ctx = ioctx + i;
              ctx->index = index;
              ctx->end = end;
              ctx->data_index = i;
              submit_io<index_t>(ctx);
              index++;
          }
        }));

    while(spdk_likely(qctx.last_index < end || qctx.num_completed < qctx.num_submitted))
    {
      spdk_nvme_qpair_process_completions(qctx.qpair, 0);
    }

    assert(qctx.num_submitted == qctx.num_completed);

    auto ticket = num_work_finished_[toggle].fetch_add(1, std::memory_order_release);
    if (1 + ticket == num_persistent_threads_launched_) {
      // The last thread signals the end of execution.
      persistent_threads_promise_[toggle]->set_value();
      ready_[toggle] = false;
    }

    toggle = (toggle + 1) % NUM_BUFFERS;
  }

  for(uint32_t i = 0; i < num_contexts_; i++) spdk_free(ioctx[i].buf);
  delete [] ioctx;
  //std::cout << "SPDK thread " << thread_id << " finished" << std::endl;
}

void SPDK::IndexSelectWorkerPersistentOverlapped(int64_t thread_id)
{
  set_affinity(thread_id + 64);

  const int64_t num_drives = spdk_util::num_drives();
  const int64_t qid = thread_id / num_drives;
  const int64_t drive_id = thread_id % num_drives;

  struct drive *drive = spdk_util::get_drive(drive_id);
  const int32_t numa_id = spdk_nvme_ctrlr_get_numa_id(drive->ctrlr);

  struct qpair_context_overlapped *qc = qctx_ovl + thread_id;
  qc->ns = drive->ns;
  qc->qpair = drive->qpairs[qid];
  for(int i = 0; i < NUM_BUFFERS; i++)
  {
    qc->ready[i] = &ready_[i];
    qc->num_submitted[i] = 0;
    qc->num_completed[i] = 0;
  }

  struct io_context_overlapped *ioctx = new struct io_context_overlapped[num_contexts_];
  for(uint32_t i = 0; i < num_contexts_; i++)
  {
      struct io_context_overlapped *ic = ioctx + i;
      ic->qctx = qc;
      ic->drive_id = drive_id;
      ic->num_drives = num_drives;
      ic->feature_size = feature_size_;
      //ic->feature_stride = featureStride;
      ic->buf = spdk_zmalloc(BLOCK_SIZE, BLOCK_SIZE, NULL, numa_id, SPDK_MALLOC_DMA);
      ic->toggle = -1; // mark unused
  }

  int toggle = 0;
  while(persistent_threads_running_)
  {
    while(persistent_threads_running_ && !ready_[toggle].load(std::memory_order_acquire)) ; // loop
    if(!persistent_threads_running_) break;
  
    AT_DISPATCH_INDEX_TYPES(
        index_[toggle].scalar_type(), "IndexSelectImpl", ([&] {
          uint64_t index = qc->last_index[toggle];
          for(uint32_t i = 0; i < num_contexts_; i++)
          {
            struct io_context_overlapped *ctx = ioctx + i;
            if(ctx->toggle >= 0) continue;

            for(; index < qc->end[toggle]; index++) {
                if(drive_id == reinterpret_cast<index_t *>(qc->node_ids[toggle])[index] % num_drives) break;
            }
            if(index >= qc->end[toggle]) { qc->last_index[toggle] = qc->end[toggle]; break; }
            ctx->index = index;
            ctx->end = qc->end[toggle];
            ctx->data_index = i;
            ctx->toggle = toggle;
            submit_io_overlapped<index_t>(ctx);
            index++;
          }
        }));

    while(spdk_likely(qc->last_index[toggle] < qc->end[toggle] || qc->num_completed[toggle] < qc->num_submitted[toggle]))
    {
      spdk_nvme_qpair_process_completions(qc->qpair, 0);
    }

    assert(qc->num_submitted[toggle] == qc->num_completed[toggle]);
    qc->num_submitted[toggle] = 0; // this immediate reset
    qc->num_completed[toggle] = 0; // seems important

    auto ticket = num_work_finished_[toggle].fetch_add(1, std::memory_order_release);
    if (1 + ticket == num_persistent_threads_launched_) {
      // The last thread signals the end of execution.
      persistent_threads_promise_[toggle]->set_value();
      ready_[toggle] = false;
    }

    toggle = (toggle + 1) % NUM_BUFFERS;
  }

  for(uint32_t i = 0; i < num_contexts_; i++) spdk_free(ioctx[i].buf);
  delete [] ioctx;
}

void SPDK::LaunchPersistentThreads()
{
  if(index_select_mode_ == LAUNCH_THREADS) return; // no persistent threads

  // modified graphbolt::_parallel_for
  std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
  num_persistent_threads_launched_ = 0;
  num_persistent_threads_finished_ = 0;
  toggle_ = 0;
  for(int i = 0; i < NUM_BUFFERS; i++) ready_[i] = false;
  persistent_threads_running_ = true;
  const int num_threads = num_qpairs_ * spdk_util::num_drives();
  for (int tid = num_threads - 1; tid >= 0; tid--) {
    if (!future_.valid()) {
      future_ = promise_.get_future();
      num_persistent_threads_launched_ = tid;
    }
    at::launch([this, &err_flag, tid] {
      try {
        if(index_select_mode_ == PERSIST_THREADS_OVERLAPPED) {
          IndexSelectWorkerPersistentOverlapped(tid);
        } else {
          IndexSelectWorkerPersistent(tid);
        }
      } catch (...) {
        if (!err_flag.test_and_set()) {
          eptr_ = std::current_exception();
        }
      }
      auto ticket = num_persistent_threads_finished_.fetch_add(1, std::memory_order_release);
      if (1 + ticket == num_persistent_threads_launched_) {
        // The last thread signals the end of execution.
        promise_.set_value();
      }
    });
  }
}

void SPDK::JoinPersistentThreads()
{
  if (num_persistent_threads_launched_ > 0) {
    persistent_threads_running_ = false;
    future_.get();
    if (eptr_) {
      std::rethrow_exception(eptr_);
    }
  }
}

}  // namespace storage
}  // namespace graphbolt
