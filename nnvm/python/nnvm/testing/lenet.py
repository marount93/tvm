# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=unused-argument
from .. import symbol as sym
from . utils import create_workload

def lenet(num_classes):
	data = sym.Variable(name='data')

	#first conv layer
	body = sym.conv2d(data=data, channels=20 , kernel_size=(5, 5), strides=(1, 1), padding=(1, 1), use_bias=False, name="conv1")
	body = sym.tanh(data=body, name='tanh1')
	body = sym.max_pool2d(data=body, pool_size=(2, 2), strides=(2, 2))

	#second conv layer
	body = sym.conv2d(data=body, channels=50 , kernel_size=(5, 5), strides=(1, 1), padding=(1, 1), use_bias=False, name="conv2")
	body = sym.tanh(data=body, name='tanh2')
	body = sym.max_pool2d(data=body, pool_size=(2, 2), strides=(2, 2))

	#first fully connected layer
	flat = sym.flatten(data=body)
	fc1 = sym.dense(data=flat, units=500, name='fc1')
	fc1 = sym.tanh(data=fc1, name='tanh3')

	#second fully connected
	fc2 = sym.dense(data=flat, units=num_classes, name='fc2')
	fc2 = sym.softmax(data=fc2, name='softmax')

	return fc2


def get_workload(batch_size=1, num_classes=10, image_shape=(1, 28, 28), dtype="float32", **kwargs):
    net = lenet(num_classes)
    return create_workload(net, batch_size, image_shape, dtype)
