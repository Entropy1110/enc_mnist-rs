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

use optee_teec::{Context, ErrorKind, Operation, ParamNone, ParamTmpRef, Session, Uuid};
use proto::{inference, Image};


pub struct InferenceTaConnector {
    sess: Session,
}

impl InferenceTaConnector {
    pub fn new(ctx: &mut Context) -> optee_teec::Result<Self> {
        let uuid = Uuid::parse_str(inference::UUID).map_err(|err| {
            println!(
                "parse uuid \"{}\" failed due to: {:?}",
                inference::UUID,
                err
            );
            ErrorKind::BadParameters
        })?;
        // Open a session with minimal data
        let dummy = [0u8; 1];
        let mut op = Operation::new(0, ParamTmpRef::new_input(&dummy), ParamNone, ParamNone, ParamNone);
        Ok(Self { sess: ctx.open_session_with_operation(uuid, &mut op)? })
    }

    pub fn begin_model_load(&mut self) -> optee_teec::Result<()> {
        let mut op = Operation::new(4, ParamNone, ParamNone, ParamNone, ParamNone);
        self.sess.invoke_command(4, &mut op)?;
        Ok(())
    }

    pub fn push_encrypted_chunk(&mut self, chunk: &[u8]) -> optee_teec::Result<()> {
        let mut op = Operation::new(5, ParamTmpRef::new_input(chunk), ParamNone, ParamNone, ParamNone);
        self.sess.invoke_command(5, &mut op)?;
        Ok(())
    }

    pub fn finalize_model_load(&mut self) -> optee_teec::Result<()> {
        let mut op = Operation::new(6, ParamNone, ParamNone, ParamNone, ParamNone);
        self.sess.invoke_command(6, &mut op)?;
        Ok(())
    }
    pub fn infer_batch(&mut self, images: &[Image]) -> optee_teec::Result<Vec<u8>> {
        let mut output = vec![0_u8; images.len()];
        let size = {
            let mut op = Operation::new(
                0,
                ParamTmpRef::new_input(bytemuck::cast_slice(images)),
                ParamTmpRef::new_output(&mut output),
                ParamNone,
                ParamNone,
            );
            self.sess.invoke_command(0, &mut op)?;
            op.parameters().1.updated_size()
        };

        if output.len() != size {
            println!("mismatch response, want {}, got {}", size, output.len());
            return Err(ErrorKind::Generic.into());
        }
        Ok(output)
    }
}

pub struct ModelEncryptorTaConnector {
    sess: Session,
}

impl ModelEncryptorTaConnector {
    pub fn new(ctx: &mut Context) -> optee_teec::Result<Self> {
        let uuid = Uuid::parse_str(inference::UUID).map_err(|err| {
            println!(
                "parse uuid \"{}\" failed due to: {:?}",
                inference::UUID,
                err
            );
            ErrorKind::BadParameters
        })?;

        // Create a dummy session just to get encryption capability
        let dummy_data = vec![0u8; 32]; // Minimal dummy data
        let mut open_op = Operation::new(
            0,
            ParamTmpRef::new_input(&dummy_data),
            ParamNone,
            ParamNone,
            ParamNone,
        );

        let sess = ctx.open_session_with_operation(uuid, &mut open_op)?;
        Ok(Self { sess })
    }

    pub fn encrypt_model(&mut self, model_data: &[u8]) -> optee_teec::Result<Vec<u8>> {
        let mut encrypted_output = vec![0_u8; model_data.len() + 1024]; // Extra space for padding
        let size = {
            let mut op = Operation::new(
                1, // Command ID for model encryption
                ParamTmpRef::new_input(model_data),
                ParamTmpRef::new_output(&mut encrypted_output),
                ParamNone,
                ParamNone,
            );
            self.sess.invoke_command(1, &mut op)?;
            op.parameters().1.updated_size()
        };

        encrypted_output.truncate(size);
        Ok(encrypted_output)
    }
}

pub struct ModelDecryptorTaConnector {
    sess: Session,
}

impl ModelDecryptorTaConnector {
    pub fn new(ctx: &mut Context) -> optee_teec::Result<Self> {
        let uuid = Uuid::parse_str(inference::UUID).map_err(|err| {
            println!(
                "parse uuid \"{}\" failed due to: {:?}",
                inference::UUID,
                err
            );
            ErrorKind::BadParameters
        })?;

        // Create a dummy session just to get decryption capability
        let dummy_data = vec![0u8; 32]; // Minimal dummy data
        let mut open_op = Operation::new(
            0,
            ParamTmpRef::new_input(&dummy_data),
            ParamNone,
            ParamNone,
            ParamNone,
        );

        let sess = ctx.open_session_with_operation(uuid, &mut open_op)?;
        Ok(Self { sess })
    }

    // pub fn decrypt_model(&mut self, encrypted_data: &[u8]) -> optee_teec::Result<Vec<u8>> {
    //     let mut decrypted_output = vec![0_u8; encrypted_data.len() + 1024]; // Extra space
    //     let size = {
    //         let mut op = Operation::new(
    //             2, // Command ID for model decryption
    //             ParamTmpRef::new_input(encrypted_data),
    //             ParamTmpRef::new_output(&mut decrypted_output),
    //             ParamNone,
    //             ParamNone,
    //         );
    //         self.sess.invoke_command(2, &mut op)?;
    //         op.parameters().1.updated_size()
    //     };

    //     decrypted_output.truncate(size);
    //     Ok(decrypted_output)
    // }
}

pub struct KeyProvisionTaConnector {
    sess: Session,
}

impl KeyProvisionTaConnector {
    pub fn new(ctx: &mut Context) -> optee_teec::Result<Self> {
        let uuid = Uuid::parse_str(inference::UUID).map_err(|err| {
            println!(
                "parse uuid \"{}\" failed due to: {:?}",
                inference::UUID,
                err
            );
            ErrorKind::BadParameters
        })?;

        // Small open to create session
        let dummy_data = vec![0u8; 16];
        let mut open_op = Operation::new(
            0,
            ParamTmpRef::new_input(&dummy_data),
            ParamNone,
            ParamNone,
            ParamNone,
        );

        let sess = ctx.open_session_with_operation(uuid, &mut open_op)?;
        Ok(Self { sess })
    }

    pub fn store_key(&mut self, key: &[u8; 32]) -> optee_teec::Result<()> {
        let mut op = Operation::new(
            3, // Command ID for key provision
            ParamTmpRef::new_input(key),
            ParamNone,
            ParamNone,
            ParamNone,
        );
        self.sess.invoke_command(3, &mut op)?;
        Ok(())
    }
}
