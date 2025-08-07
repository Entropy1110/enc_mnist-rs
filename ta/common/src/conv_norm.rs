use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig, ReLU,
    },
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct Conv2dNormActivation<B: Backend> {
    pub conv: Conv2d<B>,
    pub norm: BatchNorm<B, 2>,
    pub activation: ReLU,
}

impl<B: Backend> Conv2dNormActivation<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.norm.forward(x);
        self.activation.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct Conv2dNormActivationConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    #[config(default = "3")]
    pub kernel_size: usize,
    #[config(default = "1")]
    pub stride: usize,
    #[config(default = "1")]
    pub padding: usize,
    #[config(default = "1")]
    pub dilation: usize,
    #[config(default = "1")]
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
            dilation: 1,
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
        Conv2dNormActivation {
            conv: Conv2dConfig::new(
                [self.in_channels, self.out_channels],
                [self.kernel_size, self.kernel_size],
            )
            .with_stride([self.stride, self.stride])
            .with_padding([self.padding, self.padding])
            .with_dilation([self.dilation, self.dilation])
            .with_groups(self.groups)
            .init(device),
            norm: BatchNormConfig::new(self.out_channels).init(device),
            activation: ReLU::new(),
        }
    }
}