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

use anyhow::{anyhow, Result};
use clap::Parser;
use std::fs;
use std::time::Instant;

use aes::Aes256;
use cbc::Decryptor;
use cbc::cipher::{BlockDecryptMut, KeyIvInit, block_padding::NoPadding};
use image::EncodableLayout;

use burn::backend::ndarray::{NdArray, NdArrayDevice};

use proto::{Image, IMAGE_SIZE};
use common::Model;
use num_traits::ToPrimitive;

#[derive(serde::Deserialize)]
struct EncryptedModelFile {
    algorithm: String,
    encrypted_data: Vec<u8>,
}

#[derive(serde::Deserialize)]
struct ChunkedEncryptedModelFile {
    algorithm: String,
    chunk_size: usize,
    total_chunks: usize,
    original_size: usize,
    chunks: Vec<EncryptedChunk>,
}

#[derive(serde::Deserialize)]
struct EncryptedChunk {
    id: usize,
    size: usize,
    data: Vec<u8>,
}

#[derive(Parser, Debug)]
pub struct Args {
    /// The path of the encrypted model.
    #[arg(short, long)]
    model: String,
    /// The path of the key file.
    #[arg(short, long)]
    key: String,
    /// The path of the input binary, must be IMAGE_SIZE byte binary, can be multiple
    #[arg(short, long)]
    binary: Vec<String>,
    /// The path of the input image, must be dimension of 28x28x1 (MNIST), can be multiple
    #[arg(short, long)]
    image: Vec<String>,
}

pub fn execute(args: &Args) -> Result<()> {
    // Read key
    let key = if args.key.len() == 64 && args.key.chars().all(|c| c.is_ascii_hexdigit()) {
        hex::decode(&args.key).map_err(|e| anyhow!("Failed to decode key hex: {:?}", e))?
    } else {
        fs::read(&args.key).map_err(|e| anyhow!("Failed to read key file: {:?}", e))?
    };

    if key.len() != 32 {
        return Err(anyhow!("Invalid key size. Expected 32 bytes, got {}", key.len()));
    }

    // Read encrypted model from a json file
    let encrypted_model_content = fs::read(&args.model)?;
    
    let encrypted_model = if let Ok(chunked_model) = serde_json::from_slice::<ChunkedEncryptedModelFile>(&encrypted_model_content) {
        let mut sorted_chunks = chunked_model.chunks;
        sorted_chunks.sort_by_key(|c| c.id);
        let mut combined_data = Vec::new();
        for chunk in sorted_chunks {
            combined_data.extend_from_slice(&chunk.data);
        }
        combined_data
    } else {
        let model: EncryptedModelFile = serde_json::from_slice(&encrypted_model_content)
            .map_err(|e| anyhow!("Failed to parse encrypted model file: {:?}", e))?;
        model.encrypted_data
    };


    // Decrypt model
    let decrypt_start = Instant::now();
    if encrypted_model.len() < 16 {
        return Err(anyhow!("Invalid encrypted model length."));
    }
    let iv = &encrypted_model[..16];
    let ciphertext = &encrypted_model[16..];
    
    let cipher = Decryptor::<Aes256>::new(key.as_slice().into(), iv.into());
    let mut buf = ciphertext.to_vec();
    let decrypted_model = cipher.decrypt_padded_mut::<NoPadding>(&mut buf)
        .map_err(|e| anyhow!("Failed to decrypt model: {:?}", e))?;

    // Remove padding and length prefix
    if decrypted_model.len() < 4 {
        return Err(anyhow!("Decrypted data too short to contain length prefix. Len: {}, Data (hex): {}", 
            decrypted_model.len(), hex::encode(&decrypted_model)));
    }

    let len = u32::from_le_bytes(decrypted_model[..4].try_into().unwrap()) as usize;
    
    if len + 4 > decrypted_model.len() {
        let debug_len = std::cmp::min(16, decrypted_model.len());
        return Err(anyhow!("Invalid model length prefix. Read len: {}, Decrypted len: {}, First {} bytes (hex): {}", 
            len, decrypted_model.len(), debug_len, hex::encode(&decrypted_model[..debug_len])));
    }

    let model_data = &decrypted_model[4..4 + len];

    // Load model
    let device = NdArrayDevice::Cpu;
    let model: Model<NdArray> = Model::import(&device, model_data.to_vec())
        .map_err(|e| anyhow!("Failed to import model: {:?}", e))?;

    let decrypt_duration = decrypt_start.elapsed();
    println!("Decryption/Load Time: {} ms", decrypt_duration.as_millis());

    // Prepare input
    let mut binaries: Vec<Image> = args
        .binary
        .iter()
        .map(|v| {
            let data = fs::read(v)?;
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
            TryInto::<Image>::try_into(bytes.to_vec())
                .map_err(|err| anyhow::Error::msg(format!("cannot convert {:?} into Image", err)))
        })
        .collect::<Result<Vec<_>, anyhow::Error>>()?;
    binaries.extend(images);

    // Inference
    let infer_start = Instant::now();
    let image_to_tensor_start = Instant::now();
    let input = Model::images_to_tensors(&device, &binaries);
    let image_to_tensor_duration = image_to_tensor_start.elapsed();
    println!("Image to Tensor Conversion Time: {} ms", image_to_tensor_duration.as_millis());
    let output = model.forward(input);
    let result: Vec<u8> = output
        .iter_dim(0)
        .map(|v| {
            let data = burn::tensor::activation::softmax(v, 1);
            data.argmax(1).into_scalar().to_u8().unwrap()
        })
        .collect();
    let infer_duration = infer_start.elapsed();
    println!("Inference Time: {} ms", infer_duration.as_millis());
    println!("  (Image to Tensor: {} ms, Forward Pass: {} ms)", 
        image_to_tensor_duration.as_millis(), 
        (infer_duration - image_to_tensor_duration).as_millis());

    println!("Total Execution Time (Decryption + Inference): {} ms", (decrypt_duration + infer_duration).as_millis());

    // Print results
    for (i, binary) in args.binary.iter().enumerate() {
        println!("{}. {}: {}", i + 1, binary, result[i]);
    }

    for (i, image) in args.image.iter().enumerate() {
        println!(
            "{}. {}: {}",
            i + args.binary.len() + 1,
            image,
            result[i + args.binary.len()]
        );
    }

    Ok(())
}
