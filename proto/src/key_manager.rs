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

#![allow(dead_code)]

pub const UUID: &str = include_str!("../../../key_manager-rs/ta/uuid.txt");

pub const AES_KEY_SIZE: usize = 32;
pub const AES_BLOCK_SIZE: usize = 16;

pub const AES_KEY_OBJECT_ID: &[u8] = b"km.aes.default";
pub const RSA_KEY_OBJECT_ID: &[u8] = b"km.rsa.default";

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Command {
    GenerateAesKey = 0,
    ImportAesKey = 1,
    ExportAesKey = 2,
    EncryptAesChunk = 3,
    DecryptAesChunk = 4,
    GenerateRandom = 5,
    HasAesKey = 6,
    GenerateRsaKey = 7,
    ImportRsaKey = 8,
    ExportRsaPublic = 9,
}

impl From<u32> for Command {
    fn from(value: u32) -> Self {
        match value {
            0 => Command::GenerateAesKey,
            1 => Command::ImportAesKey,
            2 => Command::ExportAesKey,
            3 => Command::EncryptAesChunk,
            4 => Command::DecryptAesChunk,
            5 => Command::GenerateRandom,
            6 => Command::HasAesKey,
            7 => Command::GenerateRsaKey,
            8 => Command::ImportRsaKey,
            9 => Command::ExportRsaPublic,
            _ => Command::GenerateAesKey, // default fallback, caller should guard
        }
    }
}
