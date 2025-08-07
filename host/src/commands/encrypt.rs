use anyhow::Result;
use std::fs;
use std::path::Path;
use serde_json;
use clap::Args as ClapArgs;
use optee_teec::Context;

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

pub fn encrypt_model<P: AsRef<Path>>(input_path: P, output_path: P) -> Result<()> {
    println!("Encrypting model: {} -> {}", 
             input_path.as_ref().display(), 
             output_path.as_ref().display());

    let model_data = fs::read(&input_path)?;
    println!("Loaded model data: {} bytes", model_data.len());

    println!("Connecting to TA for model encryption...");
    let mut ctx = Context::new()?;
    
    println!("Requesting TA to encrypt model...");
    let encrypted_data = crate::tee::ModelEncryptorTaConnector::encrypt_model(&mut ctx, &model_data)?;
    println!("Model encrypted by TA: {} bytes", encrypted_data.len());
    
    let encrypted_model = EncryptedModelFile {
        algorithm: "AES-256-CBC".to_string(),
        encrypted_data,
    };

    let json_data = serde_json::to_vec_pretty(&encrypted_model)?;
    fs::write(&output_path, json_data)?;
    
    println!("Encrypted model saved to: {}", output_path.as_ref().display());
    println!("Host no longer has access to plaintext model!");
    Ok(())
}