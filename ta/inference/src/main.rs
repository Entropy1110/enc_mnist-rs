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

#![no_std]
#![no_main]
extern crate alloc;

use burn::{
    backend::{ndarray::NdArrayDevice, NdArray},
    tensor::cast::ToElement,
};


mod secure_storage;

use alloc::vec::Vec;
use secure_storage::{store_model_bytes, load_model_bytes, model_bytes_exists};



// No key manager; provisioning receives plaintext and stores internally.

use common::{copy_to_output, Model};
use optee_utee::{
    ta_close_session, ta_create, ta_destroy, ta_invoke_command, ta_open_session, trace_println,
};
use optee_utee::{ErrorKind, Parameters, Result};
use proto::Image;
use spin::Mutex;

type NoStdModel = Model<NdArray>;
const DEVICE: NdArrayDevice = NdArrayDevice::Cpu;
static MODEL: Mutex<Option<NoStdModel>> = Mutex::new(Option::None);
#[cfg(feature = "provision")]
static MODEL_BUF: Mutex<Vec<u8>> = Mutex::new(Vec::new());

#[ta_create]
fn create() -> Result<()> {
    trace_println!("[+] TA create");
    Ok(())
}

#[ta_open_session]
fn open_session(params: &mut Parameters) -> Result<()> {
    let mut p0 = unsafe { params.0.as_memref()? };
    let size = p0.buffer().len();
    trace_println!("[+] Open session; initial buffer size: {} bytes (ignored)", size);
    Ok(())
}

#[ta_close_session]
fn close_session() {
    trace_println!("[+] TA close session");
}

#[ta_destroy]
fn destroy() {
    trace_println!("[+] TA destroy");
}

#[ta_invoke_command]
fn invoke_command(cmd_id: u32, params: &mut Parameters) -> Result<()> {
    trace_println!("[+] TA invoke command, cmd_id: {}", cmd_id);
    
    match cmd_id {
        0 => invoke_inference(params),
        // No encrypt/decrypt/store-key commands; only provision/infer remain.
        #[cfg(feature = "provision")] 4 => invoke_begin_model_load(params),
        #[cfg(feature = "provision")] 5 => invoke_push_encrypted_chunk(params),
        #[cfg(feature = "provision")] 6 => invoke_finalize_model_load(params),
        _ => {
            trace_println!("[!] Unknown command ID: {}", cmd_id);
            Err(ErrorKind::BadParameters.into())
        }
    }
}

fn invoke_inference(params: &mut Parameters) -> Result<()> {
    trace_println!("[+] Processing inference request");
    
    trace_println!("[+] Getting input parameters...");
    let mut p0 = unsafe { params.0.as_memref()? };
    trace_println!("[+] Input buffer size: {} bytes", p0.buffer().len());
    
    trace_println!("[+] Converting to images...");
    let images: &[Image] = bytemuck::cast_slice(p0.buffer());
    trace_println!("[+] Number of images: {}", images.len());
    
    if images.is_empty() {
        trace_println!("[!] No images provided for inference");
        return Err(ErrorKind::BadParameters.into());
    }
    
    trace_println!("[+] Converting images to tensors...");
    trace_println!("[+] Image data validation - first image: {:?}", &images[0][0..8]);
    let input = NoStdModel::images_to_tensors(&DEVICE, images);
    trace_println!("[+] Tensor conversion completed");

    trace_println!("[+] Getting model from lock...");
    {
        let has_model = MODEL.lock().is_some();
        if !has_model {
            trace_println!("[+] Model not loaded in memory; checking secure storage...");
            if model_bytes_exists() {
                trace_println!("[+] Found model in secure storage; loading...");
                let plain = load_model_bytes()?;
                let imported_model = match Model::import(&DEVICE, plain) {
                    Ok(m) => m,
                    Err(_err) => {
                        trace_println!("[!] Model import from storage failed");
                        return Err(ErrorKind::BadParameters.into());
                    }
                };
                let mut mg = MODEL.lock();
                mg.replace(imported_model);
                trace_println!("[+] Model loaded from secure storage");
            } else {
                trace_println!("[!] No model in memory or secure storage");
                return Err(ErrorKind::ItemNotFound.into());
            }
        }
    }
    let model_guard = MODEL.lock();
    let model = model_guard.as_ref().ok_or(ErrorKind::CorruptObject)?;
    trace_println!("[+] Model retrieved successfully");
    
    trace_println!("[+] Running forward pass...");
    let output = model.forward(input);
    trace_println!("[+] Forward pass completed");
    
    trace_println!("[+] Processing output...");
    let result: alloc::vec::Vec<u8> = output
        .iter_dim(0)
        .map(|v| {
            let data = burn::tensor::activation::softmax(v, 1);
            data.argmax(1).into_scalar().to_u8()
        })
        .collect();
    trace_println!("[+] Output processing completed, result size: {}", result.len());

    trace_println!("[+] Copying to output...");
    copy_to_output(&mut params.1, &result)
}

// No encrypt/decrypt/store-key in this configuration.

#[cfg(feature = "provision")]
fn invoke_begin_model_load(_params: &mut Parameters) -> Result<()> {
    trace_println!("[+] Begin model load");
    let mut buf = MODEL_BUF.lock();
    buf.clear();
    Ok(())
}

#[cfg(feature = "provision")]
fn invoke_push_encrypted_chunk(params: &mut Parameters) -> Result<()> {
    let mut p0 = unsafe { params.0.as_memref()? };
    let enc = p0.buffer();
    if enc.is_empty() { return Ok(()); }
    let mut buf = MODEL_BUF.lock();
    let before = buf.len();
    // Append plaintext chunk directly (provision in plaintext)
    buf.extend_from_slice(enc);
    trace_println!("[+] Plain chunk appended: {} -> {}", before, buf.len());
    Ok(())
}

#[cfg(feature = "provision")]
fn invoke_finalize_model_load(_params: &mut Parameters) -> Result<()> {
    trace_println!("[+] Finalize model load");
    // Take accumulated plaintext buffer
    let plain = {
        let mut buf = MODEL_BUF.lock();
        core::mem::take(&mut *buf)
    };
    trace_println!("[+] Importing model with {} bytes...", plain.len());
    // Persist model bytes to secure storage
    store_model_bytes(&plain)?;
    let imported_model = match Model::import(&DEVICE, plain) {
        Ok(m) => m,
        Err(_err) => {
            trace_println!("[!] Model import failed");
            return Err(ErrorKind::BadParameters.into());
        }
    };
    let mut model = MODEL.lock();
    model.replace(imported_model);
    trace_println!("[+] Model loaded and installed");
    Ok(())
}

include!(concat!(env!("OUT_DIR"), "/user_ta_header.rs"));
