use anyhow::Result;
use clap::Args as ClapArgs;

#[derive(ClapArgs, Debug)]
pub struct Args {
    /// 32-byte AES key in hex (64 hex chars)
    #[arg(long)]
    key: String,
}

pub fn execute(args: &Args) -> Result<()> {
    let key = parse_hex_key_32(&args.key)?;
    let mut ctx = optee_teec::Context::new()?;
    let mut provisioner = crate::tee::KeyProvisionTaConnector::new(&mut ctx)?;
    provisioner.store_key(&key)?;
    println!("Secret key stored in TA secure storage.");
    Ok(())
}

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

