use alloc::{vec, vec::Vec};
use core::cmp;

use optee_utee::{
    ErrorKind, ParamIndex, Result, TaSession, TaSessionBuilder, TeeParams, Uuid,
};
use proto::key_manager::{self, Command, AES_BLOCK_SIZE, AES_KEY_SIZE};
use proto::CHUNK_SIZE;

fn with_client<F, R>(f: F) -> Result<R>
where
    F: FnOnce(&mut KeyManagerClient) -> Result<R>,
{
    let mut client = KeyManagerClient::new()?;
    f(&mut client)
}

struct KeyManagerClient {
    session: TaSession,
}

impl KeyManagerClient {
    fn new() -> Result<Self> {
        let uuid = Uuid::parse_str(key_manager::UUID.trim())?;
        let session = TaSessionBuilder::new(uuid).build()?;
        Ok(Self { session })
    }

    fn ensure_aes_key(&mut self) -> Result<()> {
        if !self.has_aes_key()? {
            self.generate_aes_key()?;
        }
        Ok(())
    }

    fn require_aes_key(&mut self) -> Result<()> {
        if !self.has_aes_key()? {
            return Err(ErrorKind::ItemNotFound.into());
        }
        Ok(())
    }

    fn generate_aes_key(&mut self) -> Result<()> {
        let mut params = TeeParams::new();
        self.session
            .invoke_command(Command::GenerateAesKey as u32, &mut params)
    }

    pub fn import_aes_key(&mut self, key: &[u8; AES_KEY_SIZE]) -> Result<()> {
        let key_buf = *key;
        let mut params = TeeParams::new().with_memref_in(ParamIndex::Arg0, &key_buf);
        self.session
            .invoke_command(Command::ImportAesKey as u32, &mut params)
    }

    pub fn export_aes_key(&mut self) -> Result<[u8; AES_KEY_SIZE]> {
        let mut buffer = [0u8; AES_KEY_SIZE];
        let mut params = TeeParams::new().with_memref_out(ParamIndex::Arg0, &mut buffer);
        self.session
            .invoke_command(Command::ExportAesKey as u32, &mut params)?;
        let written = params[ParamIndex::Arg0]
            .written_slice()
            .ok_or(ErrorKind::BadParameters)?;
        if written.len() != AES_KEY_SIZE {
            return Err(ErrorKind::BadParameters.into());
        }
        Ok(buffer)
    }

    fn has_aes_key(&mut self) -> Result<bool> {
        let mut params = TeeParams::new().with_value_out(ParamIndex::Arg0, 0, 0);
        self.session
            .invoke_command(Command::HasAesKey as u32, &mut params)?;
        let (a, _) = params[ParamIndex::Arg0]
            .output_value()
            .ok_or(ErrorKind::BadParameters)?;
        Ok(a != 0)
    }

    pub fn encrypt_data(&mut self, data: &[u8]) -> Result<Vec<u8>> {
        self.ensure_aes_key()?;
        let block_size = AES_BLOCK_SIZE;

        let mut data_with_len = Vec::with_capacity(4 + data.len());
        data_with_len.extend_from_slice(&(data.len() as u32).to_le_bytes());
        data_with_len.extend_from_slice(data);

        let padded_len = ((data_with_len.len() + block_size - 1) / block_size) * block_size;
        data_with_len.resize(padded_len, 0);

        let mut iv = self.generate_iv()?;
        let mut result = Vec::with_capacity(block_size + padded_len);
        result.extend_from_slice(&iv);

        let chunk_size = cmp::max(CHUNK_SIZE, block_size);
        let mut offset = 0;
        while offset < data_with_len.len() {
            let end = cmp::min(offset + chunk_size, data_with_len.len());
            let chunk = &data_with_len[offset..end];
            let mut encrypted_chunk = vec![0u8; chunk.len()];
            let size = self.encrypt_chunk(chunk, &mut encrypted_chunk, &mut iv)?;
            result.extend_from_slice(&encrypted_chunk[..size]);
            offset = end;
        }
        Ok(result)
    }

    pub fn decrypt_data(&mut self, encrypted: &[u8]) -> Result<Vec<u8>> {
        self.require_aes_key()?;
        if encrypted.len() < AES_BLOCK_SIZE * 2 {
            return Err(ErrorKind::BadParameters.into());
        }
        let mut iv = [0u8; AES_BLOCK_SIZE];
        iv.copy_from_slice(&encrypted[..AES_BLOCK_SIZE]);
        let ciphertext = &encrypted[AES_BLOCK_SIZE..];
        if ciphertext.len() % AES_BLOCK_SIZE != 0 {
            return Err(ErrorKind::BadParameters.into());
        }

        let chunk_size = cmp::max(CHUNK_SIZE, AES_BLOCK_SIZE);
        let mut offset = 0;
        let mut decrypted = Vec::with_capacity(ciphertext.len());
        while offset < ciphertext.len() {
            let end = cmp::min(offset + chunk_size, ciphertext.len());
            let chunk = &ciphertext[offset..end];
            let mut plain_chunk = vec![0u8; chunk.len()];
            let size = self.decrypt_chunk(chunk, &mut plain_chunk, &mut iv)?;
            decrypted.extend_from_slice(&plain_chunk[..size]);
            offset = end;
        }
        if decrypted.len() < 4 {
            return Err(ErrorKind::BadParameters.into());
        }
        let original_len = u32::from_le_bytes([
            decrypted[0],
            decrypted[1],
            decrypted[2],
            decrypted[3],
        ]) as usize;
        if original_len + 4 > decrypted.len() {
            return Err(ErrorKind::BadParameters.into());
        }
        Ok(decrypted[4..4 + original_len].to_vec())
    }

    fn encrypt_chunk(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        iv: &mut [u8; AES_BLOCK_SIZE],
    ) -> Result<usize> {
        if input.len() % AES_BLOCK_SIZE != 0 {
            return Err(ErrorKind::BadParameters.into());
        }
        let mut iv_param = *iv;
        let mut params = TeeParams::new()
            .with_memref_in(ParamIndex::Arg0, input)
            .with_memref_out(ParamIndex::Arg1, &mut output[..input.len()])
            .with_memref_inout(ParamIndex::Arg2, &mut iv_param);
        self.session
            .invoke_command(Command::EncryptAesChunk as u32, &mut params)?;
        let written = params[ParamIndex::Arg1]
            .written_slice()
            .ok_or(ErrorKind::BadParameters)?
            .len();
        let iv_slice = params[ParamIndex::Arg2]
            .written_slice()
            .ok_or(ErrorKind::BadParameters)?;
        if iv_slice.len() != AES_BLOCK_SIZE {
            return Err(ErrorKind::BadParameters.into());
        }
        let mut next_iv = [0u8; AES_BLOCK_SIZE];
        next_iv.copy_from_slice(iv_slice);
        iv.copy_from_slice(&next_iv);
        Ok(written)
    }

    fn decrypt_chunk(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        iv: &mut [u8; AES_BLOCK_SIZE],
    ) -> Result<usize> {
        if input.len() % AES_BLOCK_SIZE != 0 {
            return Err(ErrorKind::BadParameters.into());
        }
        let mut iv_param = *iv;
        let mut params = TeeParams::new()
            .with_memref_in(ParamIndex::Arg0, input)
            .with_memref_out(ParamIndex::Arg1, &mut output[..input.len()])
            .with_memref_inout(ParamIndex::Arg2, &mut iv_param);
        self.session
            .invoke_command(Command::DecryptAesChunk as u32, &mut params)?;
        let written = params[ParamIndex::Arg1]
            .written_slice()
            .ok_or(ErrorKind::BadParameters)?
            .len();
        let iv_slice = params[ParamIndex::Arg2]
            .written_slice()
            .ok_or(ErrorKind::BadParameters)?;
        if iv_slice.len() != AES_BLOCK_SIZE {
            return Err(ErrorKind::BadParameters.into());
        }
        let mut next_iv = [0u8; AES_BLOCK_SIZE];
        next_iv.copy_from_slice(iv_slice);
        iv.copy_from_slice(&next_iv);
        Ok(written)
    }

    fn generate_iv(&mut self) -> Result<[u8; AES_BLOCK_SIZE]> {
        let mut buffer = [0u8; AES_BLOCK_SIZE];
        let mut params = TeeParams::new().with_memref_inout(ParamIndex::Arg0, &mut buffer);
        self.session
            .invoke_command(Command::GenerateRandom as u32, &mut params)?;
        let written = params[ParamIndex::Arg0]
            .written_slice()
            .ok_or(ErrorKind::BadParameters)?;
        if written.len() != AES_BLOCK_SIZE {
            return Err(ErrorKind::BadParameters.into());
        }
        Ok(buffer)
    }
}

pub fn ensure_aes_key() -> Result<()> {
    with_client(|client| client.ensure_aes_key())
}

pub fn require_aes_key() -> Result<()> {
    with_client(|client| client.require_aes_key())
}

pub fn import_aes_key(key: &[u8; AES_KEY_SIZE]) -> Result<()> {
    with_client(|client| client.import_aes_key(key))
}

pub fn export_aes_key() -> Result<[u8; AES_KEY_SIZE]> {
    with_client(|client| client.export_aes_key())
}

pub fn encrypt_model_data(data: &[u8]) -> Result<Vec<u8>> {
    with_client(|client| client.encrypt_data(data))
}

pub fn decrypt_model_data(data: &[u8]) -> Result<Vec<u8>> {
    with_client(|client| client.decrypt_data(data))
}
