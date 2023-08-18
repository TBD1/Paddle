#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from test_collective_base import TestDistBase

import paddle

paddle.enable_static()


class TestCReduceMaxOp(TestDistBase):
    def _setup_config(self):
        pass

    def test_reduce_max(self):
        self.check_with_place("collective_reduce_max_op.py", "reduce_max")

    def test_reduce_calc_stream(self):
        self.check_with_place("collective_reduce_max_op_calc_stream.py", "reduce_max")


if __name__ == '__main__':
    unittest.main()
