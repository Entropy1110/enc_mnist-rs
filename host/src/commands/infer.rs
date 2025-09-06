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

use clap::Parser;
use image::EncodableLayout;
use optee_teec::Context;
use proto::{Image, IMAGE_SIZE};

#[derive(Parser, Debug)]
pub struct Args {
    /// The path of the input binary, must be IMAGE_SIZE byte binary, can be multiple
    #[arg(short, long)]
    binary: Vec<String>,
    /// The path of the input image, must be dimension of 28x28x1 (MNIST), can be multiple
    #[arg(short, long)]
    image: Vec<String>,
}

pub fn execute(args: &Args) -> anyhow::Result<()> {
    println!("Using provisioned model from TA secure storage");
    let mut ctx = Context::new()?;
    let mut caller = crate::tee::InferenceTaConnector::new(&mut ctx)?;

    let mut binaries: Vec<Image> = args
        .binary
        .iter()
        .map(|v| {
            let data = std::fs::read(v)?;
            anyhow::ensure!(data.len() == IMAGE_SIZE);

            TryInto::<Image>::try_into(data)
                .map_err(|err| anyhow::Error::msg(format!("cannot convert {:?} into Image", err)))
        })
        .collect::<Result<Vec<_>, anyhow::Error>>()?;
    let images: Vec<Image> = args
        .image
        .iter()
        .map(|v| {
            let img = image::open(v)
                .unwrap()
                .resize_exact(28, 28, image::imageops::FilterType::Triangle)
                .to_luma8();
            let bytes = img.as_bytes();
            anyhow::ensure!(bytes.len() == IMAGE_SIZE);
            TryInto::<Image>::try_into(bytes)
                .map_err(|err| anyhow::Error::msg(format!("cannot convert {:?} into Image", err)))
        })
        .collect::<Result<Vec<_>, anyhow::Error>>()?;
    binaries.extend(images);

    let result = caller.infer_batch(&binaries)?;
    anyhow::ensure!(binaries.len() == result.len());

    for (i, binary) in args.binary.iter().enumerate() {
        println!("{}. {}: {}", i + 1, binary, result[i]);
    }

    for (i, image) in args.image.iter().enumerate() {
        println!(
            "{}. {}: {}",
            i + args.binary.len() + 1,
            image,
            result[args.binary.len()]
        );
    }
    println!("Infer Success");

    Ok(())
}

// reconstruct_chunked_model removed: we never return plaintext model to host.
