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
#include <stdint.h>
#include <vector>

#include "spdk/nvme.h"

namespace spdk_util {


struct drive {
	struct spdk_nvme_ctrlr *ctrlr;
	struct spdk_nvme_ns *ns;
	struct spdk_nvme_qpair **qpairs;
};

int init(int num_qpairs, int queue_depth, const char *lcore_map);
int num_drives();
struct drive* get_drive(int drive_id);
void fini();

} // namespace spdk_util
