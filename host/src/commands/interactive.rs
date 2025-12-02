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
use anyhow::{anyhow, Result};
use std::io::{self, Write};
use std::time::{Instant, Duration};
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
}

pub fn execute(_args: &Args) -> Result<()> {
    let mut ctx_opt: Option<Context> = None;
    let mut caller_opt: Option<crate::tee::InferenceTaConnector> = None;

    fn create_session_and_caller() -> Result<(Context, crate::tee::InferenceTaConnector)> {
        let mut ctx = Context::new()?;
        let caller = crate::tee::InferenceTaConnector::new(&mut ctx)?;
        Ok((ctx, caller))
    }

    // Initial session creation
    match create_session_and_caller() {
        Ok((ctx, caller)) => {
            ctx_opt = Some(ctx);
            caller_opt = Some(caller);
        },
        Err(e) => {
            println!("Failed to create initial TA session: {:?}", e);
            return Err(e);
        }
    }

    println!("Interactive Inference Session Started.");
    println!("Commands: load <model_path>, infer <image_path>, exit");

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

        let current_caller = caller_opt.as_mut().ok_or_else(|| anyhow!("TA session not initialized."))?;
        let mut current_ctx = ctx_opt.as_mut().ok_or_else(|| anyhow!("TA Context not initialized."))?;


        match parts[0] {
            "load" => {
                if parts.len() < 2 {
                    println!("Usage: load <model_path>");
                    continue;
                }
                let model_path = parts[1];
                if let Err(e) = load_model(current_caller, model_path) {
                    println!("Error loading model: {:?}", e);
                    if format!("{:?}", e).contains("TargetDead") || format!("{:?}", e).contains("panicked") {
                        println!("TA crashed or disconnected. Attempting to reconnect...");
                        // Drop existing session and context
                        caller_opt = None;
                        ctx_opt = None;
                        match create_session_and_caller() {
                            Ok((new_ctx, new_caller)) => {
                                ctx_opt = Some(new_ctx);
                                caller_opt = Some(new_caller);
                                println!("Reconnected. Please try the 'load' command again.");
                            }
                            Err(reconnect_err) => {
                                println!("Failed to reconnect TA session: {:?}", reconnect_err);
                                println!("Please restart the application.");
                                break;
                            }
                        }
                    }
                }
            }
            "infer" => {
                if parts.len() < 2 {
                    println!("Usage: infer <image_path>");
                    continue;
                }
                let image_path = parts[1];
                if let Err(e) = infer_image(current_caller, image_path) {
                    println!("Error inferring image: {:?}", e);
                    if format!("{:?}", e).contains("TargetDead") || format!("{:?}", e).contains("panicked") {
                        println!("TA crashed or disconnected. Attempting to reconnect...");
                        // Drop existing session and context
                        caller_opt = None;
                        ctx_opt = None;
                        match create_session_and_caller() {
                            Ok((new_ctx, new_caller)) => {
                                ctx_opt = Some(new_ctx);
                                caller_opt = Some(new_caller);
                                println!("Reconnected. Please try the 'infer' command again.");
                            }
                            Err(reconnect_err) => {
                                println!("Failed to reconnect TA session: {:?}", reconnect_err);
                                println!("Please restart the application.");
                                break;
                            }
                        }
                    }
                }
            }
            "exit" => break,
            _ => println!("Unknown command: {}", parts[0]),
        }
    }

    Ok(())
}

pub fn load_model(caller: &mut crate::tee::InferenceTaConnector, model_path: &str) -> Result<Duration> {
    let data = std::fs::read(model_path)?;
    let start;
    
    if let Ok(chunked_model) = serde_json::from_slice::<ChunkedEncryptedModelFile>(&data) {
        println!("Loading chunked model...");
        caller.begin_model_load()?;
        let mut sorted_chunks = chunked_model.chunks;
        sorted_chunks.sort_by_key(|c| c.id);
        for chunk in sorted_chunks {
            caller.push_encrypted_chunk(&chunk.data)?;
        }
        start = Instant::now();
        caller.finalize_model_load()?;
    } else if let Ok(encrypted_model) = serde_json::from_slice::<EncryptedModelFile>(&data) {
        println!("Loading single encrypted model...");
        caller.begin_model_load()?;
        let data = encrypted_model.encrypted_data;
        const CHUNK: usize = 64 * 1024;
        for part in data.chunks(CHUNK) {
            caller.push_encrypted_chunk(part)?;
        }
        start = Instant::now();
        caller.finalize_model_load()?;
    } else {
        return Err(anyhow!("Unknown model format"));
    }
    let duration = start.elapsed();
    println!("TA Decryption/Load Time: {} ms", duration.as_millis());
    Ok(duration)
}

pub fn infer_image(caller: &mut crate::tee::InferenceTaConnector, image_path: &str) -> Result<(u8, Duration)> {
    let img = image::open(image_path)
        .map_err(|e| anyhow!("Failed to open image: {:?}", e))?
        .resize_exact(28, 28, image::imageops::FilterType::Triangle)
        .to_luma8();
    let bytes = img.as_bytes();
    anyhow::ensure!(bytes.len() == IMAGE_SIZE, "Invalid image size");
    let image_data: Image = bytes.try_into().map_err(|_| anyhow!("Invalid image data conversion"))?;
    
    let input = vec![image_data];
    
    let start = Instant::now();
    let result = caller.infer_batch(&input)?;
    let duration = start.elapsed();
    
    println!("TA Inference Time: {} ms", duration.as_millis());
    println!("Result: {}", result[0]);
    
    Ok((result[0], duration))
}
