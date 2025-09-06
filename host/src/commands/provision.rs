use anyhow::Result;
use clap::Args as ClapArgs;
use optee_teec::Context;

#[derive(ClapArgs, Debug)]
pub struct Args {
    /// Path to plaintext Burn model record (.bin)
    #[arg(short, long)]
    model: String,
}

pub fn execute(args: &Args) -> Result<()> {
    let model_path = std::path::absolute(&args.model)?;
    println!("Provision plaintext model from \"{}\"", model_path.display());
    let data = std::fs::read(&model_path)?;

    let mut ctx = Context::new()?;
    let mut caller = crate::tee::InferenceTaConnector::new(&mut ctx)?;

    caller.begin_model_load()?;
    const CHUNK: usize = 64 * 1024;
    for (i, part) in data.chunks(CHUNK).enumerate() {
        println!("Sending plain part {} ({} bytes)", i + 1, part.len());
        caller.push_encrypted_chunk(part)?;
    }
    caller.finalize_model_load()?;

    println!("Provision complete: model stored securely inside TA");
    Ok(())
}
