#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from op_test import OpTest

import paddle

paddle.enable_static()

# ----------------- TEST OP: rxor ---------------- #
class TestRxor(OpTest):
    def setUp(self):
        self.op_type = "__rxor__"
        self.python_api = paddle.tensor.logic.__rxor__

        self.init_dtype()
        self.init_shape()
        self.init_bound()

        x = np.random.randint(
            self.low, self.high, self.x_shape, dtype=self.dtype
        )
        y = np.random.randint(
            self.low, self.high, self.y_shape, dtype=self.dtype
        )
        out = np.bitwise_xor(y, x)

        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(
            check_cinn=True, check_pir=True, check_symbol_infer=False
        )

    def test_check_grad(self):
        pass

    def init_dtype(self):
        self.dtype = np.int32

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [2, 3, 4, 5]

    def init_bound(self):
        self.low = -100
        self.high = 100


class TestRxor_ZeroDim1(TestRxor):
    def init_shape(self):
        self.x_shape = []
        self.y_shape = []


class TestRxor_ZeroDim2(TestRxor):
    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = []


class TestRxor_ZeroDim3(TestRxor):
    def init_shape(self):
        self.x_shape = []
        self.y_shape = [2, 3, 4, 5]


class TestRxorUInt8(TestRxor):
    def init_dtype(self):
        self.dtype = np.uint8

    def init_bound(self):
        self.low = 0
        self.high = 100


class TestRxorInt8(TestRxor):
    def init_dtype(self):
        self.dtype = np.int8

    def init_shape(self):
        self.x_shape = [4, 5]
        self.y_shape = [2, 3, 4, 5]


class TestRxorInt16(TestRxor):
    def init_dtype(self):
        self.dtype = np.int16

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [4, 1]


class TestRxorInt64(TestRxor):
    def init_dtype(self):
        self.dtype = np.int64

    def init_shape(self):
        self.x_shape = [1, 4, 1]
        self.y_shape = [2, 3, 4, 5]


class TestRxorBool(TestRxor):
    def setUp(self):
        self.op_type = "__rxor__"
        self.python_api = paddle.tensor.logic.__rxor__

        self.init_shape()

        x = np.random.choice([True, False], self.x_shape)
        y = np.random.choice([True, False], self.y_shape)
        out = np.bitwise_xor(y, x)

        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out}