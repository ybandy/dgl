import time
from torch.utils.data import get_worker_info


class TimingReporter:

    def __init__(self, operation, num_minibatches, num_workers):
        self.operation = operation
        self.num_minibatches = num_minibatches
        self.num_workers = num_workers if(num_workers > 0) else 1
        self.records = []

    def report_time(self, phase, minibatch_index):
        self.records.append((phase, minibatch_index, time.time_ns()))
        if(phase == 'end'
           and minibatch_index + self.num_workers >= self.num_minibatches):
            self._write_records()

    def _write_records(self):
        worker_info = get_worker_info()
        if(worker_info is None):
            filename = f'timestamp_{self.operation}.txt'
        else:
            filename = f'timestamp_{self.operation}{worker_info.id}.txt'

        with open(filename, 'wt') as f:
            for ph, idx, ts in self.records:
                f.write(f'{self.operation}({idx}) {ph}: {ts}\n')
