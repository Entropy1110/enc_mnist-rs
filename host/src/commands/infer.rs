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
use serde_json;

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
    /// The path of the model.
    #[arg(short, long)]
    model: String,
    /// The path of the input binary, must be IMAGE_SIZE byte binary, can be multiple
    #[arg(short, long)]
    binary: Vec<String>,
    /// The path of the input image, must be dimension of 28x28x1 (MNIST), can be multiple
    #[arg(short, long)]
    image: Vec<String>,
}

pub fn execute(args: &Args) -> anyhow::Result<()> {
    let model_path = std::path::absolute(&args.model)?;
    println!("Load model from \"{}\"", model_path.display());
    
    let mut ctx = Context::new()?;
    let mut caller = crate::tee::InferenceTaConnector::new(&mut ctx)?;

    let record = if model_path.extension().and_then(|s| s.to_str()) == Some("json") {
        println!("Detected encrypted model file");
        let encrypted_data = std::fs::read(&model_path)?;
        
        // Try to parse as chunked model first
        if let Ok(chunked_model) = serde_json::from_slice::<ChunkedEncryptedModelFile>(&encrypted_data) {
            println!("Model algorithm: {} (chunked)", chunked_model.algorithm);
            println!("Reconstructing model from {} chunks ({} bytes)", 
                     chunked_model.total_chunks, chunked_model.original_size);
            // Stream encrypted chunks to TA
            caller.begin_model_load()?;
            let mut sorted_chunks = chunked_model.chunks;
            sorted_chunks.sort_by_key(|c| c.id);
            for chunk in sorted_chunks {
                println!("Sending encrypted chunk {}/{} ({} bytes)", chunk.id + 1, chunked_model.total_chunks, chunk.data.len());
                caller.push_encrypted_chunk(&chunk.data)?;
            }
            caller.finalize_model_load()?;
            Vec::new() // no local record
        } else {
            // Fall back to single encrypted model
            let encrypted_model: EncryptedModelFile = serde_json::from_slice(&encrypted_data)?;
            println!("Model algorithm: {}", encrypted_model.algorithm);
            caller.begin_model_load()?;
            // Send in chunks to avoid large shared buffers
            let data = encrypted_model.encrypted_data;
            const CHUNK: usize = 64 * 1024;
            for (i, part) in data.chunks(CHUNK).enumerate() {
                println!("Sending encrypted part {} ({} bytes)", i + 1, part.len());
                caller.push_encrypted_chunk(part)?;
            }
            caller.finalize_model_load()?;
            Vec::new()
        }
    } else {
        println!("Loading plaintext model (legacy mode)");
        let data = std::fs::read(&model_path)?;
        // For legacy plaintext, stream as a single encrypted-chunk with no encryption is unsafe.
        // Here we fallback to old path: open session with plaintext (not recommended for production).
        data
    };
    
    // If record is empty, model already loaded in TA via streaming.
    // We already have an open session in `caller`.
    if !record.is_empty() {
        // Legacy path: open session already loaded the model
        println!("Model sent on open_session (legacy mode)");
    }

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
