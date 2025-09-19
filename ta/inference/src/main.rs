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


mod key_manager;
mod secure_storage;

use alloc::vec::Vec;
use key_manager::KeyManager;
use secure_storage::{store_ta_aes_key, load_ta_aes_key, ta_aes_key_exists, export_ta_aes_key};



static mut KEY_MANAGER: Option<KeyManager> = None;

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
        #[cfg(feature = "encrypt-model")]
        1 => invoke_encrypt_model(params),
        // 2 => invoke_decrypt_model(params),
        3 => invoke_store_key(params),
        4 => invoke_begin_model_load(params),
        5 => invoke_push_encrypted_chunk(params),
        6 => invoke_finalize_model_load(params),
        7 => invoke_export_aes_key(params),
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

#[cfg(feature = "encrypt-model")]
fn invoke_encrypt_model(params: &mut Parameters) -> Result<()> {
    trace_println!("[+] Processing model encryption request");
    
    let mut p0 = unsafe { params.0.as_memref()? };
    let mut p1 = unsafe { params.1.as_memref()? };
    
    let model_data = p0.buffer();
    trace_println!("[+] Received model data: {} bytes", model_data.len());
    
    let aes_key = if ta_aes_key_exists() {
        trace_println!("[+] Loading existing TA AES key for encryption");
        load_ta_aes_key()?
    } else {
        trace_println!("[+] Generating new TA AES key for encryption");
        let new_key = KeyManager::generate_aes_key()?;
        store_ta_aes_key(&new_key)?;
        new_key
    };
    
    let mut key_manager = KeyManager::new(aes_key)?;
    
    trace_println!("[+] Encrypting model with TA AES key...");
    let encrypted_model = key_manager.encrypt_data(model_data)?;
    trace_println!("[+] Model encrypted, size: {} bytes", encrypted_model.len());
    
    if p1.buffer().len() < encrypted_model.len() {
        trace_println!("[!] Output buffer too small: {} < {}", p1.buffer().len(), encrypted_model.len());
        return Err(ErrorKind::ShortBuffer.into());
    }
    
    p1.buffer()[..encrypted_model.len()].copy_from_slice(&encrypted_model);
    p1.set_updated_size(encrypted_model.len());
    
    trace_println!("[+] Encrypted model returned to host");
    Ok(())
}


fn invoke_store_key(params: &mut Parameters) -> Result<()> {
    trace_println!("[+] Processing key provision request");
    let mut p0 = unsafe { params.0.as_memref()? };
    let key_buf = p0.buffer();
    if key_buf.len() != 32 {
        trace_println!("[!] Invalid key size: {}", key_buf.len());
        return Err(ErrorKind::BadParameters.into());
    }
    let mut key = [0u8; 32];
    key.copy_from_slice(key_buf);
    store_ta_aes_key(&key)?;
    unsafe {
        KEY_MANAGER = Some(KeyManager::new(key)?);
    }
    trace_println!("[+] Secret key stored in secure storage");
    Ok(())
}

fn ensure_key_manager() -> Result<()> {
    if !ta_aes_key_exists() {
        return Err(ErrorKind::ItemNotFound.into());
    }
    if unsafe { KEY_MANAGER.is_none() } {
        let aes_key = load_ta_aes_key()?;
        unsafe { KEY_MANAGER = Some(KeyManager::new(aes_key)?) };
    }
    Ok(())
}

fn invoke_export_aes_key(params: &mut Parameters) -> Result<()> {
    trace_println!("[+] Export AES key request received");
    let key = export_ta_aes_key()?;
    let mut p0 = unsafe { params.0.as_memref()? };
    if p0.buffer().len() < key.len() {
        trace_println!("[!] Output buffer too small for AES key");
        return Err(ErrorKind::ShortBuffer.into());
    }
    p0.buffer()[..key.len()].copy_from_slice(&key);
    p0.set_updated_size(key.len());
    Ok(())
}

fn invoke_begin_model_load(_params: &mut Parameters) -> Result<()> {
    trace_println!("[+] Begin model load");
    ensure_key_manager()?;
    let mut buf = MODEL_BUF.lock();
    buf.clear();
    Ok(())
}

fn invoke_push_encrypted_chunk(params: &mut Parameters) -> Result<()> {
    let mut p0 = unsafe { params.0.as_memref()? };
    let enc = p0.buffer();
    if enc.is_empty() { return Ok(()); }
    let mut buf = MODEL_BUF.lock();
    let before = buf.len();
    // Append encrypted bytes as-is; decrypt once at finalize
    buf.extend_from_slice(enc);
    trace_println!("[+] Encrypted chunk appended: {} -> {}", before, buf.len());
    Ok(())
}

fn invoke_finalize_model_load(_params: &mut Parameters) -> Result<()> {
    trace_println!("[+] Finalize model load");
    // Decrypt full encrypted buffer once
    ensure_key_manager()?;
    let encrypted = {
        let mut buf = MODEL_BUF.lock();
        core::mem::take(&mut *buf)
    };
    let key_manager = unsafe { KEY_MANAGER.as_mut().ok_or(ErrorKind::BadState)? };
    trace_println!("[+] Decrypting accumulated encrypted model: {} bytes", encrypted.len());
    let plain = key_manager.decrypt_data(&encrypted)?;
    trace_println!("[+] Decrypted model size: {} bytes", plain.len());
    trace_println!("[+] Importing model with {} bytes...", plain.len());
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
