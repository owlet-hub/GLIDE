# =============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

# Use this variable to update RAPIDS and RAFT versions
set(RAPIDS_VERSION "24.08")

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/RAFT_RAPIDS.cmake)
    file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-${RAPIDS_VERSION}/RAPIDS.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/RAFT_RAPIDS.cmake)
endif()
include(${CMAKE_CURRENT_BINARY_DIR}/RAFT_RAPIDS.cmake)