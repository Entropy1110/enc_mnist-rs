// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (
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
use std::time::Duration;
use std::env;
use std::path::PathBuf;
use std::fs;
use optee_teec::Context;
use burn::backend::ndarray::NdArrayDevice;

use crate::commands::{interactive, interactive_local};
use crate::tee::{InferenceTaConnector, KeyProvisionTaConnector, ModelEncryptorTaConnector};

#[derive(Parser, Debug)]
pub struct Args {
    /// The path of the plaintext model to encrypt.
    #[arg(short, long, default_value = "../model/model.bin")]
    plaintext_model: String,
    /// The path of the encrypted model (output of encryption, input for tests).
    #[arg(short, long, default_value = "encmodel.json")]
    model: String,
    /// The path of the input image.
    #[arg(short, long, default_value = "samples/7.png")]
    image: String,
}

#[derive(serde::Serialize)]
struct EncryptedModelFile {
    algorithm: String,
    encrypted_data: Vec<u8>,
}

const VALID_KEY: &str = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
const INVALID_KEY: &str = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeefe";

pub fn execute(args: &Args) -> Result<()> {
    let exe_path = env::current_exe()?;
    println!("Running tests directly within the process: {}", exe_path.display());

    let plaintext_model_path = PathBuf::from(&args.plaintext_model);
    let model_path_str = &args.model;
    let image_path_str = &args.image;
    let model_path = PathBuf::from(model_path_str);
    let image_path = PathBuf::from(image_path_str);

    if !plaintext_model_path.exists() {
        return Err(anyhow!("Plaintext model file not found: {}", plaintext_model_path.display()));
    }
    if !image_path.exists() {
        return Err(anyhow!("Image file not found: {}", image_path.display()));
    }

    // 0. Provision Key & Encrypt Model
    println!("\n--- [Setup] Provisioning Key & Encrypting Model ---");
    provision_key(VALID_KEY)?;
    println!("Key provisioned successfully.");

    encrypt_model(&plaintext_model_path, &model_path)?;
    println!("Model encrypted and saved to: {}", model_path.display());

    // 2. Functional Test: Key Management (Local)
    test_key_management(model_path_str, VALID_KEY, INVALID_KEY)?;

    // 3. Functional Test: Session Persistence (TA)
    // 4. Non-functional Test: Performance (TA vs Local)
    let (ta_load_time, ta_infer_times) = test_session_persistence_and_perf_ta(model_path_str, image_path_str)?;
    let (local_load_time, local_infer_times, local_conv_times) = test_perf_local(model_path_str, image_path_str, VALID_KEY)?;

    // 5. Analysis Report
    print_analysis_report(ta_load_time, &ta_infer_times, local_load_time, &local_infer_times, &local_conv_times);

    Ok(())
}

fn provision_key(key_hex: &str) -> Result<()> {
    let mut ctx = Context::new()?;
    let mut key_provision = KeyProvisionTaConnector::new(&mut ctx)?;
    let key_bytes = hex::decode(key_hex)?;
    if key_bytes.len() != 32 {
        return Err(anyhow!("Invalid key length"));
    }
    let mut key_arr = [0u8; 32];
    key_arr.copy_from_slice(&key_bytes);
    key_provision.store_key(&key_arr)?;
    Ok(())
}

fn encrypt_model(input_path: &PathBuf, output_path: &PathBuf) -> Result<()> {
    let mut ctx = Context::new()?;
    let mut encryptor = ModelEncryptorTaConnector::new(&mut ctx)?;
    let plaintext_data = fs::read(input_path)?;
    
    let encrypted_data = encryptor.encrypt_model(&plaintext_data)
        .map_err(|e| anyhow!("TA Encryption failed: {:?}", e))?;

    let model_file = EncryptedModelFile {
        algorithm: "AES-256-CBC".to_string(),
        encrypted_data,
    };

    let json = serde_json::to_string_pretty(&model_file)?;
    fs::write(output_path, json)?;
    Ok(())
}

fn test_key_management(model_path: &str, valid_key: &str, invalid_key: &str) -> Result<()> {
    println!("\n--- [Functional Test] Key Management (Local) ---");
    let device = NdArrayDevice::Cpu;

    // Test Invalid Key
    println!("Testing Invalid Key...");
    let result = interactive_local::load_model(model_path, invalid_key, &device);
    if result.is_err() {
        println!("PASS: Load failed with invalid key as expected.");
    } else {
        println!("FAIL: Load succeeded or no error reported with invalid key.");
        return Err(anyhow!("Key Management Test Failed (Invalid Key)"));
    }

    // Test Valid Key
    println!("Testing Valid Key...");
    let result = interactive_local::load_model(model_path, valid_key, &device);
    if result.is_ok() {
        println!("PASS: Load succeeded with valid key.");
    } else {
        println!("FAIL: Load failed with valid key: {:?}", result.err());
        return Err(anyhow!("Key Management Test Failed (Valid Key)"));
    }

    println!("Result: Key Management Verification PASSED.");
    Ok(())
}

fn test_session_persistence_and_perf_ta(model_path: &str, image_path: &str) -> Result<(u64, Vec<u64>)> {
    println!("\n--- [Functional & Performance Test] Session Persistence & Performance (TA) ---");
    let mut ctx = Context::new()?;
    let mut caller = InferenceTaConnector::new(&mut ctx)?;

    // Load Model
    let load_time_duration = interactive::load_model(&mut caller, model_path)?;
    let load_time = load_time_duration.as_millis() as u64;

    // Infer multiple times
    let mut infer_times = Vec::new();
    for _ in 0..10 {
        let (_, duration) = interactive::infer_image(&mut caller, image_path)?;
        infer_times.push(duration.as_millis() as u64);
    }

    // If we reached here without error, persistence is verified
    println!("PASS: Session Persistence verified (Multiple inferences in one session).");

    Ok((load_time, infer_times))
}

fn test_perf_local(model_path: &str, image_path: &str, key: &str) -> Result<(u64, Vec<u64>, Vec<u64>)> {
    println!("\n--- [Performance Test] Local Performance ---");
    let device = NdArrayDevice::Cpu;

    // Load Model
    let (model, load_duration) = interactive_local::load_model(model_path, key, &device)?;
    let load_time = load_duration.as_millis() as u64;

    // Infer multiple times
    let mut infer_times = Vec::new();
    let mut conv_times = Vec::new();

    for _ in 0..10 {
        let (_, infer_duration, conv_duration) = interactive_local::infer_image(&model, image_path, &device)?;
        infer_times.push(infer_duration.as_millis() as u64);
        conv_times.push(conv_duration.as_millis() as u64);
    }

    Ok((load_time, infer_times, conv_times))
}

fn print_analysis_report(
    ta_load_time: u64, ta_infer_times: &[u64],
    local_load_time: u64, local_infer_times: &[u64], local_conv_times: &[u64]
) {
    println!("\n============================================================");
    println!("                FINAL ANALYSIS REPORT                       ");
    println!("============================================================");

    println!("\n1. Cold Start Latency (Decryption/Load Time)");
    println!("------------------------------------------------------------");
    println!("TA Load Time    : {} ms", ta_load_time);
    println!("Local Load Time : {} ms", local_load_time);
    if local_load_time > 0 {
        let ratio = ta_load_time as f64 / local_load_time as f64;
        println!("Comparison      : TA is {:.1}x slower than Local", ratio);
    }
    println!("Analysis        : Security world encryption overhead and memory copy are major factors.");

    println!("\n2. Inference Throughput & Latency");
    println!("------------------------------------------------------------");
    if !ta_infer_times.is_empty() {
        println!("TA Inference    : Initial {} ms -> Avg (subsequent) {:.1} ms", 
            ta_infer_times[0], 
            ta_infer_times.iter().skip(1).sum::<u64>() as f64 / (ta_infer_times.len() - 1).max(1) as f64
        );
    }
    if !local_infer_times.is_empty() {
        println!("Local Inference : Initial {} ms -> Avg (subsequent) {:.1} ms", 
            local_infer_times[0], 
            local_infer_times.iter().skip(1).sum::<u64>() as f64 / (local_infer_times.len() - 1).max(1) as f64
        );
    }
    println!("Analysis        : Local optimization vs World Switch overhead analysis required based on values.");

    println!("\n3. Resource Efficiency (Image to Tensor Conversion - Local)");
    println!("------------------------------------------------------------");
    if !local_conv_times.is_empty() {
        println!("Conversion Time : Initial {} ms -> Subsequent {:?}", local_conv_times[0], &local_conv_times[1..]);
        println!("Analysis        : Check for OS page cache effects (subsequent times should be near 0).");
    }
    
    println!("\n============================================================");
    println!("Test Complete.");
}
