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
use alloc::vec;
use core::cmp::max;

use burn::{
    prelude::*,
    config::Config,
    module::Module,
    nn::{
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig,
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig,
    },
    tensor::{backend::Backend, Tensor, TensorData, activation::relu},
    record::{FullPrecisionSettings, Recorder, RecorderError},
};
use proto::{Image, IMAGE_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS};

const INVERTED_RESIDUAL_SETTINGS: [[usize; 4]; 7] = [
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
];
const ROUND_NEAREST: usize = 8;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    features: Vec<ConvBlock<B>>,
    classifier: Classifier<B>,
    avg_pool: AdaptiveAvgPool2d,
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device) -> Self {
        let config = MobileNetV2Config::new();
        config.init(device)
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut x = input;
        for layer in &self.features {
            match layer {
                ConvBlock::InvertedResidual(block) => {
                    x = block.forward(&x);
                }
                ConvBlock::Conv(conv) => {
                    x = conv.forward(x);
                }
            }
        }
        x = self.avg_pool.forward(x);
        let x = x.flatten(1, 3);
        self.classifier.forward(x)
    }

    pub fn import(device: &B::Device, record: Vec<u8>) -> Result<Self, RecorderError> {
        let recorder = burn::record::BinBytesRecorder::<FullPrecisionSettings>::new();
        let record = recorder.load(record, device)?;

        let m = Self::new(device);
        Ok(m.load_record(record))
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
enum ConvBlock<B: Backend> {
    InvertedResidual(InvertedResidual<B>),
    Conv(Conv2dNormActivation<B>),
}

#[derive(Module, Debug)]
struct Classifier<B: Backend> {
    dropout: Dropout,
    linear: Linear<B>,
}

impl<B: Backend> Classifier<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.dropout.forward(input);
        self.linear.forward(x)
    }
}

#[derive(Debug, Config)]
pub struct MobileNetV2Config {
    #[config(default = "1000")]
    num_classes: usize,
    #[config(default = "1.0")]
    width_mult: f32,
    #[config(default = "0.2")]
    dropout: f64,
}

impl MobileNetV2Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let input_channel = 32;
        let last_channel = 1280;

        let make_divisible = |v, divisor| {
            let new_v = (v + divisor as f32 / 2.0) as usize / divisor * divisor;
            let mut new_v = max(new_v, divisor);
            if (new_v as f32) < 0.9 * v {
                new_v += divisor;
            }
            new_v
        };

        let mut input_channel =
            make_divisible(input_channel as f32 * self.width_mult, ROUND_NEAREST);
        let last_channel = make_divisible(
            last_channel as f32 * f32::max(1.0, self.width_mult),
            ROUND_NEAREST,
        );

        let mut features = vec![ConvBlock::Conv(
            Conv2dNormActivationConfig::new(3, input_channel)
                .with_kernel_size(3)
                .with_stride(2)
                .init(device),
        )];
        
        for [t, c, n, s] in INVERTED_RESIDUAL_SETTINGS.into_iter() {
            let output_channel = make_divisible(c as f32 * self.width_mult, ROUND_NEAREST);
            for i in 0..n {
                let stride = if i == 0 { s } else { 1 };
                features.push(ConvBlock::InvertedResidual(
                    InvertedResidualConfig::new(input_channel, output_channel, stride, t)
                        .init(device),
                ));
                input_channel = output_channel;
            }
        }
        
        features.push(ConvBlock::Conv(
            Conv2dNormActivationConfig::new(input_channel, last_channel)
                .with_kernel_size(1)
                .init(device),
        ));

        let classifier = Classifier {
            dropout: DropoutConfig::new(self.dropout).init(),
            linear: LinearConfig::new(last_channel, self.num_classes).init(device),
        };

        Model {
            features,
            classifier,
            avg_pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct Conv2dNormActivation<B: Backend> {
    pub conv: Conv2d<B>,
    pub norm: BatchNorm<B, 2>,
}

impl<B: Backend> Conv2dNormActivation<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.norm.forward(x);
        relu(x)
    }
}

#[derive(Debug)]
pub struct Conv2dNormActivationConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub groups: usize,
}

impl Conv2dNormActivationConfig {
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            groups: 1,
        }
    }

    pub fn with_kernel_size(mut self, kernel_size: usize) -> Self {
        self.kernel_size = kernel_size;
        if kernel_size == 1 {
            self.padding = 0;
        }
        self
    }

    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    pub fn with_groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Conv2dNormActivation<B> {
        let padding = if self.padding > 0 {
            burn::nn::PaddingConfig2d::Explicit(self.padding, self.padding)
        } else {
            burn::nn::PaddingConfig2d::Valid
        };

        Conv2dNormActivation {
            conv: Conv2dConfig::new(
                [self.in_channels, self.out_channels],
                [self.kernel_size, self.kernel_size],
            )
            .with_stride([self.stride, self.stride])
            .with_padding(padding)
            .with_groups(self.groups)
            .init(device),
            norm: BatchNormConfig::new(self.out_channels).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct InvertedResidual<B: Backend> {
    use_res_connect: bool,
    pw: Option<Conv2dNormActivation<B>>,
    dw: Conv2dNormActivation<B>,
    pw_linear: ConvNorm<B>,
}

impl<B: Backend> InvertedResidual<B> {
    pub fn forward(&self, input: &Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = input.clone();
        
        if let Some(ref pw) = self.pw {
            x = pw.forward(x);
        }
        
        x = self.dw.forward(x);
        x = self.pw_linear.forward(x);
        
        if self.use_res_connect {
            x + input.clone()
        } else {
            x
        }
    }
}

#[derive(Module, Debug)]
struct ConvNorm<B: Backend> {
    conv: Conv2d<B>,
    norm: BatchNorm<B, 2>,
}

impl<B: Backend> ConvNorm<B> {
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        self.norm.forward(x)
    }
}

#[derive(Debug)]
pub struct InvertedResidualConfig {
    pub inp: usize,
    pub oup: usize,
    pub stride: usize,
    pub expand_ratio: usize,
}

impl InvertedResidualConfig {
    pub fn new(inp: usize, oup: usize, stride: usize, expand_ratio: usize) -> Self {
        Self {
            inp,
            oup,
            stride,
            expand_ratio,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> InvertedResidual<B> {
        let use_res_connect = self.stride == 1 && self.inp == self.oup;
        let hidden_dim = self.inp * self.expand_ratio;

        let pw = if self.expand_ratio != 1 {
            Some(
                Conv2dNormActivationConfig::new(self.inp, hidden_dim)
                    .with_kernel_size(1)
                    .init(device),
            )
        } else {
            None
        };

        let dw = Conv2dNormActivationConfig::new(hidden_dim, hidden_dim)
            .with_kernel_size(3)
            .with_stride(self.stride)
            .with_groups(hidden_dim)
            .init(device);

        let pw_linear = ConvNorm {
            conv: Conv2dConfig::new([hidden_dim, self.oup], [1, 1]).init(device),
            norm: BatchNormConfig::new(self.oup).init(device),
        };

        InvertedResidual {
            use_res_connect,
            pw,
            dw,
            pw_linear,
        }
    }
}

impl<B: Backend> Model<B> {

    pub fn image_to_tensor(device: &B::Device, image: &Image) -> Tensor<B, 4> {
        let tensor_data = TensorData::new(
            image.iter().map(|&x| x as f32 / 255.0).collect::<Vec<f32>>(),
            [1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
        );
        let tensor = Tensor::from_data(tensor_data.convert::<B::FloatElem>(), device);
        tensor.permute([0, 3, 1, 2])
    }

    pub fn images_to_tensors(device: &B::Device, images: &[Image]) -> Tensor<B, 4> {
        let batch_size = images.len();
        let mut data = Vec::with_capacity(batch_size * IMAGE_SIZE);
        
        for image in images {
            for &pixel in image.iter() {
                data.push(pixel as f32 / 255.0);
            }
        }
        
        let tensor_data = TensorData::new(
            data,
            [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
        );
        let tensor = Tensor::from_data(tensor_data.convert::<B::FloatElem>(), device);
        tensor.permute([0, 3, 1, 2])
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
