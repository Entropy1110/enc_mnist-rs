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

mod commands;
mod tee;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Infer(commands::infer::Args),
    #[cfg(feature = "encrypt-model")]
    EncryptModel(commands::encrypt::Args),
    StoreKey(commands::store_key::Args),
    #[cfg(feature = "encrypt-model")]
    VerifyModel(commands::verify_model::Args),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Infer(args) => commands::infer::execute(&args),
        #[cfg(feature = "encrypt-model")]
        Commands::EncryptModel(args) => commands::encrypt::execute(&args),
        Commands::StoreKey(args) => commands::store_key::execute(&args),
        #[cfg(feature = "encrypt-model")]
        Commands::VerifyModel(args) => commands::verify_model::execute(&args),
    }
}
