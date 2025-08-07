# README.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is `enc_mnist-rs`, an encrypted MNIST machine learning inference system built on OP-TEE TrustZone. It demonstrates secure machine learning where models are encrypted and processed entirely within the Trusted Execution Environment (TEE), preventing the host from accessing plaintext models.

## Architecture

### Two-World Design
- **REE (Rich Execution Environment)**: Host application in `host/` handles CLI, file I/O, and communication with TEE
- **TEE (Trusted Execution Environment)**: Trusted Application (TA) in `ta/inference/` performs all cryptographic operations and model inference

### Current Security Model (Enterprise-Grade DEK/KEK)
- **KEK (Key Encryption Key)**: Derived from TA UUID using simplified digest approach
- **DEK (Data Encryption Key)**: Generated per model, encrypted with KEK, stored in secure storage
- **Envelope Encryption**: Models encrypted with DEK, DEK encrypted with KEK
- **Zero-Access**: Host never sees plaintext models after encryption

### Core Components
- `host/src/commands/encrypt.rs`: Model encryption command (calls TA for encryption)
- `host/src/commands/infer.rs`: Inference command with encrypted model support
- `ta/inference/src/key_manager.rs`: HSM-based key management (KEK derivation, DEK encryption)
- `ta/inference/src/secure_storage.rs`: Persistent storage for encrypted DEKs
- `ta/inference/src/main.rs`: TA main with dual command support (cmd_id 0=infer, cmd_id 1=encrypt)

## Build Commands

### Build All Components
```bash
make all           # Build both host and TA components
make toolchain     # Install Rust toolchain
make host          # Build host application only
make ta            # Build TA (inference) only
make clean         # Clean all build artifacts
```

### Host Application Usage
```bash
# Encrypt a model (host calls TA for encryption)
./enc_mnist-rs encrypt-model --input ../samples/model.bin --output ./model_enc.json

# Perform inference with encrypted model
./enc_mnist-rs infer -m ./model_enc.json -i ../samples/7.png

# Legacy serve command (uses plaintext models)
./enc_mnist-rs serve -m ../samples/model.bin -p 3000
```

## Key Files to Understand

### Protocol Definition
- `proto/src/inference.rs`: Shared data structures between REE and TEE
- `proto/src/lib.rs`: Protocol exports

### Host Components
- `host/src/main.rs`: CLI parser with subcommands (encrypt-model, infer, serve)
- `host/src/tee.rs`: REE-TEE communication layer with session management
- `host/src/encrypt.rs`: AES encryption utilities (fallback for legacy operations)

### TA Components
- `ta/inference/src/main.rs`: TA entry point with invoke_command routing
- `ta/inference/build.rs`: TA build configuration
- `ta/inference/uuid.txt`: TA unique identifier (ff09aa8a-fbb9-4734-ae8c-d7cd1a3f6744)

### Common Libraries
- `ta/common/src/model.rs`: Burn ML framework model definitions
- `ta/common/src/utils.rs`: Shared utilities between TAs

## Development Workflow

1. **Model Encryption**: Use `encrypt-model` command to have TA encrypt models and store securely
2. **Testing**: Use provided samples in `host/samples/` (0.bin through 9.bin, images, model.bin)
3. **Debugging**: Check TA logs through OP-TEE for TEE-side issues
4. **Security**: All cryptographic operations must occur in TA, never expose plaintext models to host

## Planned Simplification: AES Key Storage

The current enterprise-grade DEK/KEK system will be simplified to store simple AES keys directly in secure storage:

### Changes Required
1. **Replace key_manager.rs**: Remove KEK derivation and DEK encryption, implement direct AES key storage
2. **Simplify secure_storage.rs**: Store raw AES keys instead of encrypted DEKs
3. **Update main.rs**: Remove envelope encryption logic, use stored AES keys directly
4. **Modify host encrypt.rs**: Remove DEK generation, let TA generate and store AES keys
5. **Update protocol**: Remove key metadata from encrypted model format

### Benefits of Simplification
- Reduced memory usage in TEE (no KEK derivation or envelope encryption)
- Simpler key lifecycle management
- Direct AES operations without intermediate encryption layers
- Maintain zero-access security model (host still cannot access keys or plaintext models)

## Security Notes

- All AES operations use OP-TEE's internal optee_utee modules (no external crypto libraries in TA)
- Models are encrypted as: `original_size` (u32) + `encrypted_data` with AES-CBC-NOPAD
- Fixed IV used for demonstration (production should use random IVs)
- TA UUID serves as hardware-bound key material for KEK derivation
- Secure storage objects are bound to specific TA instance

## Testing

Use the provided sample files:
- Images: `host/samples/7.png` (28x28 handwritten digit)
- Binaries: `host/samples/0.bin` through `host/samples/9.bin` (784-byte raw data)
- Model: `host/samples/model.bin` (pre-trained MNIST model)

Expected workflow:
1. Encrypt model: 31575 bytes → 31584 bytes (with padding)
2. Inference: Should correctly identify digit "7" from sample image
