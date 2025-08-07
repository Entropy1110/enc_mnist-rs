use anyhow::Result;
use std::fs;
use std::path::Path;
use serde_json;
use clap::Args as ClapArgs;
use optee_teec::Context;
use burn::{
    backend::NdArray,
    tensor::{backend::Backend, Device},
    record::{FullPrecisionSettings, Recorder},
    module::Module,
};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};

#[cfg(feature = "encrypt-model")]
use common::Model;

#[derive(ClapArgs)]
pub struct Args {
    #[arg(long)]
    input: String,
    
    #[arg(long)]
    output: String,
}

pub fn execute(args: &Args) -> Result<()> {
    encrypt_model(&args.input, &args.output)
}

#[derive(serde::Serialize)]
struct EncryptedModelFile {
    algorithm: String,
    encrypted_data: Vec<u8>,
}

#[derive(serde::Serialize)]
struct ChunkedEncryptedModelFile {
    algorithm: String,
    chunk_size: usize,
    total_chunks: usize,
    original_size: usize,
    chunks: Vec<EncryptedChunk>,
}

#[derive(serde::Serialize)]
struct EncryptedChunk {
    id: usize,
    size: usize,
    data: Vec<u8>,
}

pub fn encrypt_model<P: AsRef<Path>>(input_path: P, output_path: P) -> Result<()> {
    println!("Encrypting model: {} -> {}", 
             input_path.as_ref().display(), 
             output_path.as_ref().display());

    let input_path_ref = input_path.as_ref();
    let model_data = if let Some(extension) = input_path_ref.extension() {
        if extension == "pth" {
            #[cfg(feature = "encrypt-model")]
            {
                println!("Detected PyTorch model file (.pth)");
                convert_pytorch_to_burn_binary(input_path_ref)?
            }
            #[cfg(not(feature = "encrypt-model"))]
            {
                anyhow::bail!("PyTorch model support requires 'encrypt-model' feature");
            }
        } else {
            println!("Loading as Burn binary model");
            fs::read(&input_path)?
        }
    } else {
        println!("No extension detected, loading as Burn binary model");
        fs::read(&input_path)?
    };
    
    println!("Model data prepared: {} bytes", model_data.len());

    // Use chunked encryption for large models (>1MB)
    if model_data.len() > 1024 * 1024 {
        encrypt_model_chunked(&model_data, &output_path)?;
    } else {
        println!("Connecting to TA for model encryption...");
        let mut ctx = Context::new()?;
        let mut encryptor = crate::tee::ModelEncryptorTaConnector::new(&mut ctx)?;
        
        println!("Requesting TA to encrypt model...");
        let encrypted_data = encryptor.encrypt_model(&model_data)?;
        println!("Model encrypted by TA: {} bytes", encrypted_data.len());
        
        let encrypted_model = EncryptedModelFile {
            algorithm: "AES-256-CBC".to_string(),
            encrypted_data,
        };

        let json_data = serde_json::to_vec_pretty(&encrypted_model)?;
        fs::write(&output_path, json_data)?;
    }
    
    println!("Encrypted model saved to: {}", output_path.as_ref().display());
    println!("Host no longer has access to plaintext model!");
    Ok(())
}

#[cfg(feature = "encrypt-model")]
fn convert_pytorch_to_burn_binary(pytorch_path: &Path) -> Result<Vec<u8>> {
    println!("Converting PyTorch model to Burn binary format...");
    
    let device: Device<NdArray> = Default::default();
    
    // Use common Model structure (same as TA)
    let empty_model = Model::<NdArray>::new(&device);
    
    // Load PyTorch weights with key remapping for MobileNetV2
    let load_args = LoadArgs::new(pytorch_path.to_path_buf())
        .with_key_remap("features\\.(0|18)\\.0.(.+)", "features.$1.conv.$2")
        .with_key_remap("features\\.(0|18)\\.1.(.+)", "features.$1.norm.$2")
        .with_key_remap("features\\.1\\.conv.0.0.(.+)", "features.1.dw.conv.$1")
        .with_key_remap("features\\.1\\.conv.0.1.(.+)", "features.1.dw.norm.$1")
        .with_key_remap("features\\.1\\.conv.1.(.+)", "features.1.pw_linear.conv.$1")
        .with_key_remap("features\\.1\\.conv.2.(.+)", "features.1.pw_linear.norm.$1")
        .with_key_remap("features\\.([2-9]|1[0-7])\\.conv.0.0.(.+)", "features.$1.pw.conv.$2")
        .with_key_remap("features\\.([2-9]|1[0-7])\\.conv.0.1.(.+)", "features.$1.pw.norm.$2")
        .with_key_remap("features\\.([2-9]|1[0-7])\\.conv.1.0.(.+)", "features.$1.dw.conv.$2")
        .with_key_remap("features\\.([2-9]|1[0-7])\\.conv.1.1.(.+)", "features.$1.dw.norm.$2")
        .with_key_remap("features\\.([2-9]|1[0-7])\\.conv.2.(.+)", "features.$1.pw_linear.conv.$2")
        .with_key_remap("features\\.([2-9]|1[0-7])\\.conv.3.(.+)", "features.$1.pw_linear.norm.$2")
        .with_key_remap("classifier.1.(.+)", "classifier.linear.$1");
    
    println!("Loading PyTorch weights...");
    let record = PyTorchFileRecorder::<FullPrecisionSettings>::new().load(load_args, &device)?;
    
    // Apply weights to model
    let model_with_weights = empty_model.load_record(record);
    
    // Serialize to Burn binary format
    let recorder = burn::record::BinBytesRecorder::<FullPrecisionSettings>::new();
    let binary_data = recorder.record(model_with_weights.into_record(), ())?;
    
    println!("PyTorch model converted to Burn binary: {} bytes", binary_data.len());
    Ok(binary_data)
}

fn encrypt_model_chunked<P: AsRef<Path>>(model_data: &[u8], output_path: P) -> Result<()> {
    use proto::CHUNK_SIZE;
    
    println!("Using chunked encryption for large model ({} bytes)", model_data.len());
    println!("Chunk size: {} bytes", CHUNK_SIZE);
    
    let mut ctx = Context::new()?;
    let mut encryptor = crate::tee::ModelEncryptorTaConnector::new(&mut ctx)?;
    let mut encrypted_chunks = Vec::new();
    
    // Split model data into chunks
    let chunks: Vec<&[u8]> = model_data.chunks(CHUNK_SIZE).collect();
    let total_chunks = chunks.len();
    
    println!("Processing {} chunks...", total_chunks);
    
    for (i, chunk) in chunks.iter().enumerate() {
        println!("Encrypting chunk {}/{} ({} bytes)", i + 1, total_chunks, chunk.len());
        
        let encrypted_chunk_data = encryptor.encrypt_model(chunk)?;
        
        encrypted_chunks.push(EncryptedChunk {
            id: i,
            size: chunk.len(),
            data: encrypted_chunk_data,
        });
        
        println!("Chunk {}/{} encrypted successfully", i + 1, total_chunks);
    }
    
    let chunked_encrypted_model = ChunkedEncryptedModelFile {
        algorithm: "AES-256-CBC-Chunked".to_string(),
        chunk_size: CHUNK_SIZE,
        total_chunks,
        original_size: model_data.len(),
        chunks: encrypted_chunks,
    };
    
    let json_data = serde_json::to_vec_pretty(&chunked_encrypted_model)?;
    fs::write(&output_path, json_data)?;
    
    println!("Chunked encrypted model saved: {} chunks, {} bytes total", 
             total_chunks, model_data.len());
    Ok(())
}

