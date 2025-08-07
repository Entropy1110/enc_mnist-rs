# README.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is `enc_mnist-rs`, an encrypted MNIST machine learning inference system built on OP-TEE TrustZone. It demonstrates secure machine learning where models are encrypted and processed entirely within the Trusted Execution Environment (TEE), preventing the host from accessing plaintext models.

## Architecture

### Two-World Design
- **REE (Rich Execution Environment)**: Host application in `host/` handles CLI, file I/O, and communication with TEE
- **TEE (Trusted Execution Environment)**: Trusted Application (TA) in `ta/inference/` performs all cryptographic operations and model inference

### Current Security Model (Simplified AES Key Storage)
- **AES Master Key**: Random 256-bit key generated per TA instance
- **Secure Storage**: Master key stored in OP-TEE secure storage, bound to TA
- **Direct Encryption**: Models encrypted directly with AES-CBC using random IVs
- **Zero-Access**: Host never sees plaintext models or encryption keys

### Core Components
- `host/src/commands/encrypt.rs`: Model encryption command (calls TA for encryption)
- `host/src/commands/infer.rs`: Inference command with encrypted model support
- `ta/inference/src/key_manager.rs`: AES key management with secure random IV generation
- `ta/inference/src/secure_storage.rs`: Persistent storage for AES master keys
- `ta/inference/src/main.rs`: TA main with dual command support (cmd_id 0=infer, cmd_id 1=encrypt)

## Build Commands

### Build All Components
```bash
make all           # Build both host and TA components (with encrypt-model feature)
make no-encrypt    # Build without encrypt-model feature (production mode)
make toolchain     # Install Rust toolchain
make host          # Build host application only
make ta            # Build TA (inference) only
make clean         # Clean all build artifacts
```

### Feature-Specific Builds
```bash
# Build with specific features
make FEATURES="encrypt-model" all

# Build without default features
make NO_FEATURES="--no-default-features" all
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

## Conditional Compilation Features

The project supports conditional compilation to include or exclude the `encrypt-model` functionality:

### Available Features
- **encrypt-model**: Enables model encryption capabilities (included by default)
  - Adds `encrypt-model` CLI command to host application
  - Includes encryption handler in TA (cmd_id 1)
  - Required for model provisioning workflows

### Feature Benefits
- **Production Builds**: Use `make no-encrypt` to remove encryption code and reduce binary size
- **Development/Provisioning**: Use `make all` to include full encryption capabilities
- **Security**: Prevents accidental inclusion of encryption features in production deployments

## Security Notes

- All AES operations use OP-TEE's internal optee_utee modules (no external crypto libraries in TA)
- Models encrypted as: `[Random IV (16 bytes)] + [AES-CBC encrypted data]`
- **Random IV Generation**: Each encryption uses cryptographically secure random IV via `Random::generate()`
- **Proper OP-TEE API Usage**: Follows recommended cipher initialization and finalization patterns
- **Memory Safety**: Fixed IV extraction using owned arrays instead of borrowed slices
- **Key Isolation**: AES master keys stored in OP-TEE secure storage, inaccessible to host
- Secure storage objects are bound to specific TA instance

## Testing

Use the provided sample files:
- Images: `host/samples/7.png` (28x28 handwritten digit)
- Binaries: `host/samples/0.bin` through `host/samples/9.bin` (784-byte raw data)
- Model: `host/samples/model.bin` (pre-trained MNIST model)

Expected workflow:
1. Encrypt model: 31575 bytes → 31591 bytes (16-byte IV + padded encrypted data)
2. Inference: Should correctly identify digit "7" from sample image

### Security Improvements Applied
- **Fixed IV Vulnerability**: Replaced fixed `[0u8; 16]` IV with random generation
- **Memory Safety**: Fixed slice borrowing issues causing TA panics
- **API Compliance**: Proper OP-TEE Cipher API usage with correct initialization
