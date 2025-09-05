// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use alloc::vec::Vec;
use burn::{
    prelude::*,
    record::{FullPrecisionSettings, Recorder, RecorderError},
    tensor::{backend::Backend, Tensor, TensorData},
};
use proto::{Image, IMAGE_SIZE, NUM_CLASSES};

/// Enhanced multi-layer neural network model for MNIST classification
#[derive(Module, Debug)]
pub struct MnistModel<B: Backend> {
    linear1: nn::Linear<B>,
    linear2: nn::Linear<B>,
    linear3: nn::Linear<B>,
    output: nn::Linear<B>,
    dropout: nn::Dropout,
}

impl<B: Backend> MnistModel<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            linear1: nn::LinearConfig::new(IMAGE_SIZE, 512).init(device),
            linear2: nn::LinearConfig::new(512, 256).init(device),
            linear3: nn::LinearConfig::new(256, 128).init(device),
            output: nn::LinearConfig::new(128, NUM_CLASSES).init(device),
            dropout: nn::DropoutConfig::new(0.5).init(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(input);
        let x = burn::tensor::activation::relu(x);
        let x = self.dropout.forward(x);

        let x = self.linear2.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.dropout.forward(x);

        let x = self.linear3.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.dropout.forward(x);

        self.output.forward(x)
    }

    pub fn export(&self) -> Result<Vec<u8>, RecorderError> {
        let recorder = burn::record::BinBytesRecorder::<FullPrecisionSettings>::new();
        recorder.record(self.clone().into_record(), ())
    }

    pub fn import(device: &B::Device, record: Vec<u8>) -> Result<Self, RecorderError> {
        let recorder = burn::record::BinBytesRecorder::<FullPrecisionSettings>::new();
        let record = recorder.load(record, device)?;

        let m = Self::new(device);
        Ok(m.load_record(record))
    }
}

// A wrapper to be compatible with records exported from a unified model
// where MNIST submodel is stored under a `mnist` field.
#[derive(Module, Debug)]
pub struct UnifiedModel<B: Backend> {
    mnist: MnistModel<B>,
}

impl<B: Backend> UnifiedModel<B> {
    pub fn new(device: &B::Device) -> Self {
        Self { mnist: MnistModel::new(device) }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.mnist.forward(input)
    }

    pub fn export(&self) -> Result<Vec<u8>, RecorderError> {
        let recorder = burn::record::BinBytesRecorder::<FullPrecisionSettings>::new();
        recorder.record(self.clone().into_record(), ())
    }

    pub fn import(device: &B::Device, bytes: Vec<u8>) -> Result<Self, RecorderError> {
        let recorder = burn::record::BinBytesRecorder::<FullPrecisionSettings>::new();
        match recorder.load(bytes.clone(), device) {
            Ok(record) => {
                let m = Self::new(device);
                Ok(m.load_record(record))
            }
            Err(_) => {
                // Fallback: try loading as plain MnistModel then wrap
                let mnist = MnistModel::import(device, bytes)?;
                Ok(Self { mnist })
            }
        }
    }

    pub fn image_to_tensor(device: &B::Device, image: &Image) -> Tensor<B, 2> {
        MnistModel::<B>::image_to_tensor(device, image)
    }

    pub fn images_to_tensors(device: &B::Device, images: &[Image]) -> Tensor<B, 2> {
        MnistModel::<B>::images_to_tensors(device, images)
    }

    pub fn labels_to_tensors(device: &B::Device, labels: &[u8]) -> Tensor<B, 1, Int> {
        MnistModel::<B>::labels_to_tensors(device, labels)
    }
}

// Keep existing name `Model` for compatibility with TA/host code.
pub type Model<B> = UnifiedModel<B>;

impl<B: Backend> MnistModel<B> {
    // Originally inspired by the burn/examples/mnist-inference-web package.
    pub fn image_to_tensor(device: &B::Device, image: &Image) -> Tensor<B, 2> {
        let tensor = TensorData::from(image.as_slice()).convert::<B::FloatElem>();
        let tensor = Tensor::<B, 1>::from_data(tensor, device);
        let tensor = tensor.reshape([1, IMAGE_SIZE]);

        // Normalize input: [0,1] then standardize with mean/std from PyTorch MNIST
        ((tensor / 255) - 0.1307) / 0.3081
    }

    pub fn images_to_tensors(device: &B::Device, images: &[Image]) -> Tensor<B, 2> {
        let tensors = images
            .iter()
            .map(|v| Self::image_to_tensor(device, v))
            .collect();
        Tensor::cat(tensors, 0)
    }

    pub fn labels_to_tensors(device: &B::Device, labels: &[u8]) -> Tensor<B, 1, Int> {
        let targets = labels
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data([(*item as i64).elem::<B::IntElem>()], device)
            })
            .collect();
        Tensor::cat(targets, 0)
    }
}
