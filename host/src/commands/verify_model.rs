use anyhow::Result;
use clap::Args as ClapArgs;

#[derive(ClapArgs, Debug)]
pub struct Args {
    /// Path to plaintext Burn record (.bin) to verify with burn 0.17 loader
    #[arg(long)]
    input: String,
}

pub fn execute(args: &Args) -> Result<()> {
    use burn::{backend::NdArray, prelude::*};
    let device: <NdArray as Backend>::Device = Default::default();
    let bytes = std::fs::read(&args.input)?;
    println!("Verifying Burn record with burn 0.17 loader: {} bytes", bytes.len());
    let _model = common::Model::<NdArray>::import(&device, bytes)?;
    println!("Model record is compatible with burn 0.17 (TA loader)");
    Ok(())
}

