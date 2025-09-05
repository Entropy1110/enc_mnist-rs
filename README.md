# README.md

## Overview

`enc_mnist-rs` is an encrypted MNIST inference example on OP‑TEE TrustZone. The host never handles plaintext model weights; instead, it streams an encrypted model to the TA, which decrypts and imports it inside the TEE before performing inference.

## Architecture

### Two-World Design
- **REE (Rich Execution Environment)**: Host application in `host/` handles CLI, file I/O, and communication with TEE
- **TEE (Trusted Execution Environment)**: Trusted Application (TA) in `ta/inference/` performs all cryptographic operations and model inference

### Security Model (Key Provision + Encrypted Streaming)
- **Per-TA AES Key (256‑bit)**: Provisioned from host via CLI and stored in OP‑TEE Trusted Storage.
- **Encrypted Model at Rest/In Transit**: Model file is AES‑256‑CBC encrypted outside the TA (host‑side) using the provisioned key. The host only ever holds ciphertext + IV.
- **Streaming to TA**: The encrypted model is streamed in chunks to the TA; the TA decrypts once on finalize and imports the model inside TEE memory.
- **Zero Plaintext on Host**: Plaintext model is never reconstructed on the host.

### Core Components
- `proto/`: Shared no‑std types and TA UUID (28×28×1, 10 classes).
- `ta/common/src/model.rs`: Burn MNIST MLP model (784→512→256→128→10) and import helpers.
- `ta/inference/src/main.rs`: TA entry; commands for key store and streaming model load (begin/push/finalize) and inference.
- `ta/inference/src/key_manager.rs`: AES‑256‑CBC (random IV), decrypt/encrypt helpers; Trusted Storage integration.
- `host/src/commands/store_key.rs`: Provision 32‑byte key (hex) to TA Trusted Storage.
- `host/src/commands/encrypt.rs`: Encrypt plaintext model on host with provided key (IV||ciphertext JSON).
- `host/src/commands/infer.rs`: Stream encrypted model JSON to TA, then run inference.
- `host/src/commands/verify_model.rs`: Verify plaintext model record is compatible with TA loader (burn 0.17).

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
# 1) Provision the TA key (32 bytes hex = 64 chars)
./enc_mnist-rs store-key --key 00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff

# 2) Encrypt plaintext Burn record on host with the same key
./enc_mnist-rs encrypt-model \
  --input ./model_mnist.bin \
  --output ./model_enc.json \
  --key 00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff

# 3) Inference (streams encrypted model to TA, then infers)
./enc_mnist-rs infer --model ./model_enc.json -i ./samples/7.png

# (Optional) Verify plaintext model record is compatible with TA loader (burn 0.17)
./enc_mnist-rs verify-model --input ./model_mnist.bin
```

## Key Files to Understand

### Protocol Definition
- `proto/src/inference.rs`: Shared data structures between REE and TEE
- `proto/src/lib.rs`: Protocol exports

### Host Components
- `host/src/main.rs`: CLI parser with subcommands (store-key, encrypt-model, infer, verify-model)
- `host/src/tee.rs`: REE↔TEE connector; implements streaming model load APIs
- `host/src/commands/encrypt.rs`: Host‑side AES‑256‑CBC encryption of model
- `host/src/commands/infer.rs`: Encrypted model streaming + inference
- `host/src/commands/store_key.rs`: Key provisioning to TA
- `host/src/commands/verify_model.rs`: Burn 0.17 import verification for plaintext records

### TA Components
- `ta/inference/src/main.rs`: TA entry, commands: 0=infer, 1=encrypt (optional), 2=decrypt, 3=store-key, 4=begin-load, 5=push-chunk, 6=finalize-load
- `ta/inference/build.rs`: TA memory sizes (data 32MiB, stack 8MiB, framework stack 16MiB)
- `ta/inference/uuid.txt`: TA UUID

### Common Libraries
- `ta/common/src/model.rs`: Burn ML framework model definitions
- `ta/common/src/utils.rs`: Shared utilities between TAs

## Development Workflow

1. Train/export plaintext Burn model (recommended Burn 0.17 for compatibility).
2. Provision key to TA: `store-key --key <64-hex>`.
3. Encrypt model on host: `encrypt-model --input <bin> --output model_enc.json --key <64-hex>`.
4. Inference: `infer --model model_enc.json -i samples/7.png` (host streams encrypted model; TA decrypts+imports inside TEE).
5. Troubleshoot format using `verify-model` (host tries TA loader on plaintext file).

## Conditional Compilation Features

The project supports conditional compilation to include or exclude the `encrypt-model` functionality:

### Available Features
- **encrypt-model** (default): Enables host `encrypt-model` command and TA-side encrypt/decrypt helpers (cmd 1/2). For production, you can `make no-encrypt` to reduce code surface.

### Feature Benefits
- **Production Builds**: Use `make no-encrypt` to remove encryption code and reduce binary size
- **Development/Provisioning**: Use `make all` to include full encryption capabilities
- **Security**: Prevents accidental inclusion of encryption features in production deployments

## Security Notes

- Model file format: `IV (16 bytes) || AES‑CBC(ciphertext)`; plaintext begins with a 4‑byte LE length prefix used to remove zero padding precisely after decrypt.
- TA decrypts using the provisioned key from Trusted Storage; the host only handles ciphertext.
- Encrypted streaming: the host sends encrypted chunks; the TA concatenates and decrypts once on finalize (CBC requires single pass with IV).
- Keys remain inside TEE; provisioning writes to OP‑TEE Trusted Storage bound to the TA UUID.

## Testing

- Samples: `host/samples/7.png` (28×28), `host/samples/{0..9}.bin` (784 bytes).
- Proto constants: IMAGE_WIDTH=28, IMAGE_HEIGHT=28, IMAGE_CHANNELS=1, NUM_CLASSES=10.
- Verify plaintext record format quickly on host: `verify-model --input <bin>`.
- Expect the encrypted JSON to be slightly larger than plaintext (IV + block alignment to 16 bytes).

### Notes
- TA uses Burn 0.17 (no‑std, ndarray). Ensure plaintext model was exported with Burn 0.17 for best compatibility.
- The legacy HTTP serve command has been removed to keep the surface minimal and avoid plaintext paths.
