use optee_utee::{AlgorithmId, AttributeId, AttributeMemref, Cipher, OperationMode, TransientObject, TransientObjectType, ErrorKind, Result, Random};

pub struct KeyManager {
    aes_key: [u8; 32],
    cipher: Option<Cipher>,
    key_object: Option<TransientObject>,
}

impl KeyManager {
    pub fn new(aes_key: [u8; 32]) -> Result<Self> {
        Ok(Self { 
            aes_key,
            cipher: None,
            key_object: None,
        })
    }

    pub fn generate_aes_key() -> Result<[u8; 32]> {
        let mut key = [0u8; 32];
        Random::generate(&mut key);
        Ok(key)
    }


    pub fn encrypt_data(&mut self, data: &[u8]) -> Result<alloc::vec::Vec<u8>> {
        let block_size = 16;
        let padded_len = ((data.len() + block_size - 1) / block_size) * block_size;
        let mut padded_data = data.to_vec();
        padded_data.resize(padded_len, 0);

        self.init_encrypt_cipher()?;
        
        // Generate random IV for each encryption
        let mut iv = [0u8; 16];
        Random::generate(&mut iv);
        
        if let Some(ref mut cipher) = self.cipher {
            cipher.init(&iv);
            let mut encrypted = alloc::vec![0u8; padded_len];
            let encrypted_len = cipher.update(&padded_data, &mut encrypted)?;
            encrypted.truncate(encrypted_len);
            
            // Prepend IV to encrypted data
            let mut result = alloc::vec![0u8; 16 + encrypted_len];
            result[0..16].copy_from_slice(&iv);
            result[16..].copy_from_slice(&encrypted);
            Ok(result)
        } else {
            Err(ErrorKind::BadState.into())
        }
    }

    pub fn decrypt_data(&mut self, encrypted_data: &[u8]) -> Result<alloc::vec::Vec<u8>> {
        // Check minimum length (IV + at least one block)
        if encrypted_data.len() < 32 {
            return Err(ErrorKind::BadParameters.into());
        }
        
        self.init_decrypt_cipher()?;
        
        // Extract IV from first 16 bytes - copy to fixed array
        let mut iv = [0u8; 16];
        iv.copy_from_slice(&encrypted_data[0..16]);
        let ciphertext = &encrypted_data[16..];
        
        if let Some(ref mut cipher) = self.cipher {
            cipher.init(&iv);
            let mut decrypted = alloc::vec![0u8; ciphertext.len()];
            let decrypted_len = cipher.update(ciphertext, &mut decrypted)?;
            decrypted.truncate(decrypted_len);
            Ok(decrypted)
        } else {
            Err(ErrorKind::BadState.into())
        }
    }
    
    fn init_encrypt_cipher(&mut self) -> Result<()> {
        let mut key_object = TransientObject::allocate(TransientObjectType::Aes, 256)?;
        let cipher = Cipher::allocate(AlgorithmId::AesCbcNopad, OperationMode::Encrypt, 256)?;
        
        let attr = AttributeMemref::from_ref(AttributeId::SecretValue, &self.aes_key);
        key_object.populate(&[attr.into()])?;
        cipher.set_key(&key_object)?;
        
        self.key_object = Some(key_object);
        self.cipher = Some(cipher);
        Ok(())
    }
    
    fn init_decrypt_cipher(&mut self) -> Result<()> {
        let mut key_object = TransientObject::allocate(TransientObjectType::Aes, 256)?;
        let cipher = Cipher::allocate(AlgorithmId::AesCbcNopad, OperationMode::Decrypt, 256)?;
        
        let attr = AttributeMemref::from_ref(AttributeId::SecretValue, &self.aes_key);
        key_object.populate(&[attr.into()])?;
        cipher.set_key(&key_object)?;
        
        self.cipher = Some(cipher);
        Ok(())
    }
}