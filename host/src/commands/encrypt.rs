use anyhow::Result;
use clap::Args as ClapArgs;
use rand::RngCore;
use serde_json;
use std::fs;
use std::path::Path;

#[derive(ClapArgs)]
pub struct Args {
    #[arg(long)]
    input: String,
    
    #[arg(long)]
    output: String,

    /// 32-byte AES key in hex (64 hex chars)
    #[arg(long)]
    key: String,
}

pub fn execute(args: &Args) -> Result<()> {
    encrypt_model(&args.input, &args.output, &args.key)
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

pub fn encrypt_model<P: AsRef<Path>>(input_path: P, output_path: P, key_hex: &str) -> Result<()> {
    println!("Encrypting model: {} -> {}", 
             input_path.as_ref().display(), 
             output_path.as_ref().display());

    let model_data = fs::read(&input_path)?;
    println!("Model data prepared: {} bytes", model_data.len());

    // Key from CLI (hex string)
    let key_bytes = parse_hex_key_32(key_hex)?;

    // Encrypt on host using provided key
    let encrypted_data = encrypt_with_key_host(&key_bytes, &model_data)?;
    println!("Model encrypted on host: {} bytes", encrypted_data.len());

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

// Note: MobileNetV2 / PyTorch .pth conversion removed. Provide Burn binary (.bin).

fn parse_hex_key_32(hex_str: &str) -> Result<[u8; 32]> {
    let s = hex_str.trim();
    if s.len() != 64 {
        anyhow::bail!("Key must be 64 hex chars (32 bytes)");
    }
    let mut key = [0u8; 32];
    for i in 0..32 {
        let byte_str = &s[i * 2..i * 2 + 2];
        key[i] = u8::from_str_radix(byte_str, 16)
            .map_err(|_| anyhow::anyhow!("Invalid hex at position {}", i))?;
    }
    Ok(key)
}

fn encrypt_with_key_host(key: &[u8; 32], data: &[u8]) -> Result<Vec<u8>> {
    use aes::Aes256;
    use cbc::cipher::{block_padding::NoPadding, BlockEncryptMut, KeyIvInit};
    type Aes256CbcEnc = cbc::Encryptor<Aes256>;

    // Build plaintext: [len:4][data][zero padding to 16 bytes]
    let block = 16usize;
    let orig_len = data.len();
    let mut plaintext = Vec::with_capacity(4 + orig_len + block);
    plaintext.extend_from_slice(&(orig_len as u32).to_le_bytes());
    plaintext.extend_from_slice(data);
    let pad_len = (block - (plaintext.len() % block)) % block;
    if pad_len > 0 {
        plaintext.extend(std::iter::repeat(0u8).take(pad_len));
    }

    // Random IV
    let mut iv = [0u8; 16];
    rand::rng().fill_bytes(&mut iv);

    let mut buf = plaintext.clone();
    // CBC-NOPAD style (buffer must be block aligned)
    let encrypted = Aes256CbcEnc::new(key.into(), (&iv).into())
        .encrypt_padded_mut::<NoPadding>(&mut buf, plaintext.len())
        .map_err(|_| anyhow::anyhow!("CBC encryption failed"))?;

    // Output: IV || ciphertext
    let mut out = Vec::with_capacity(16 + encrypted.len());
    out.extend_from_slice(&iv);
    out.extend_from_slice(encrypted);
    Ok(out)
}
