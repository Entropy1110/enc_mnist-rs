use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig, ReLU6,
    },
    tensor::{backend::Backend, Tensor},
};

use super::conv_norm::{Conv2dNormActivation, Conv2dNormActivationConfig};

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

#[derive(Config, Debug)]
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