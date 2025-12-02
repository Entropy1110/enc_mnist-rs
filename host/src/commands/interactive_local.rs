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
use std::time::{Instant, Duration};
use std::io::{self, Write};

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
}

pub fn execute(_args: &Args) -> Result<()> {
    let mut loaded_model: Option<Model<NdArray>> = None;
    let device = NdArrayDevice::Cpu;

    println!("Interactive Local Inference Session Started.");
    println!("Commands: load <model_path> <key>, infer <image_path>, exit");

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            break;
        }

        let parts: Vec<&str> = input.trim().split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "load" => {
                if parts.len() < 3 {
                    println!("Usage: load <model_path> <key>");
                    continue;
                }
                let model_path = parts[1];
                let key_input = parts[2];
                
                match load_model(model_path, key_input, &device) {
                    Ok((model, _duration)) => {
                        loaded_model = Some(model);
                        println!("Model loaded successfully.");
                    }
                    Err(e) => println!("Error loading model: {:?}", e),
                }
            }
            "infer" => {
                if parts.len() < 2 {
                    println!("Usage: infer <image_path>");
                    continue;
                }
                let image_path = parts[1];
                
                if let Some(model) = &loaded_model {
                    if let Err(e) = infer_image(model, image_path, &device) {
                        println!("Error inferring image: {:?}", e);
                    }
                } else {
                    println!("No model loaded. Use 'load' command first.");
                }
            }
            "exit" => break,
            _ => println!("Unknown command: {}", parts[0]),
        }
    }

    Ok(())
}

pub fn load_model(model_path: &str, key_input: &str, device: &NdArrayDevice) -> Result<(Model<NdArray>, Duration)> {
    // Read key
    let key = if key_input.len() == 64 && key_input.chars().all(|c| c.is_ascii_hexdigit()) {
        hex::decode(key_input).map_err(|e| anyhow!("Failed to decode key hex: {:?}", e))?
    } else {
        fs::read(key_input).map_err(|e| anyhow!("Failed to read key file: {:?}", e))?
    };

    if key.len() != 32 {
        return Err(anyhow!("Invalid key size. Expected 32 bytes, got {}", key.len()));
    }

    // Read encrypted model from a json file
    let encrypted_model_content = fs::read(model_path)?;
    
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
        return Err(anyhow!("Decrypted data too short to contain length prefix."));
    }
    let len = u32::from_le_bytes(decrypted_model[..4].try_into().unwrap()) as usize;
    if len + 4 > decrypted_model.len() {
        return Err(anyhow!("Invalid model length prefix."));
    }
    let model_data = &decrypted_model[4..4 + len];

    // Load model
    let model: Model<NdArray> = Model::import(device, model_data.to_vec())
        .map_err(|e| anyhow!("Failed to import model: {:?}", e))?;

    let decrypt_duration = decrypt_start.elapsed();
    println!("Decryption/Load Time: {} ms", decrypt_duration.as_millis());

    Ok((model, decrypt_duration))
}

pub fn infer_image(model: &Model<NdArray>, image_path: &str, device: &NdArrayDevice) -> Result<(u8, Duration, Duration)> {
    let img = image::open(image_path)
        .map_err(|e| anyhow!("Failed to open image: {:?}", e))?
        .resize_exact(28, 28, image::imageops::FilterType::Triangle)
        .to_luma8();
    let bytes = img.as_bytes();
    anyhow::ensure!(bytes.len() == IMAGE_SIZE, "Invalid image size");
    let image_data: Image = bytes.try_into().map_err(|_| anyhow!("Invalid image data conversion"))?;
    
    let binaries = vec![image_data];

    // Inference
    let infer_start = Instant::now();
    let image_to_tensor_start = Instant::now();
    let input = Model::images_to_tensors(device, &binaries);
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

    println!("Result: {}", result[0]);
    
    Ok((result[0], infer_duration, image_to_tensor_duration))
}
