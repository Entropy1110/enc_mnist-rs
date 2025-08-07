use optee_utee::{DataFlag, ObjectStorageConstants, PersistentObject, GenericObject, Result, ErrorKind};

const TA_AES_KEY_ID: &[u8] = b"ta_unique_aes_key";

pub fn store_ta_aes_key(aes_key: &[u8; 32]) -> Result<()> {
    let obj_data_flag = DataFlag::ACCESS_READ
        | DataFlag::ACCESS_WRITE
        | DataFlag::ACCESS_WRITE_META
        | DataFlag::OVERWRITE;

    let mut init_data: [u8; 0] = [];
    let mut obj_id = TA_AES_KEY_ID.to_vec();

    match PersistentObject::create(
        ObjectStorageConstants::Private,
        &mut obj_id,
        obj_data_flag,
        None,
        &mut init_data,
    ) {
        Ok(mut object) => {
            object.write(aes_key)?;
            Ok(())
        }
        Err(e) => Err(e),
    }
}

pub fn load_ta_aes_key() -> Result<[u8; 32]> {
    let mut obj_id = TA_AES_KEY_ID.to_vec();

    match PersistentObject::open(
        ObjectStorageConstants::Private,
        &mut obj_id,
        DataFlag::ACCESS_READ,
    ) {
        Ok(object) => {
            let obj_info = object.info()?;
            if obj_info.data_size() != 32 {
                return Err(ErrorKind::BadParameters.into());
            }
            
            let mut buffer = [0u8; 32];
            let read_bytes = object.read(&mut buffer)?;
            
            if read_bytes != 32 {
                return Err(ErrorKind::BadParameters.into());
            }
            
            Ok(buffer)
        }
        Err(e) => Err(e),
    }
}

pub fn ta_aes_key_exists() -> bool {
    let mut obj_id = TA_AES_KEY_ID.to_vec();
    
    PersistentObject::open(
        ObjectStorageConstants::Private,
        &mut obj_id,
        DataFlag::ACCESS_READ,
    ).is_ok()
}