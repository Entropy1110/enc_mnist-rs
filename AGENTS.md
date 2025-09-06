# Repository Guidelines

## Project Structure & Modules
- `host/`: REE CLI binary `enc_mnist-rs` with subcommands `infer` and `encrypt-model` (feature‑gated).
- `ta/inference/`: OP‑TEE Trusted Application (TA) performing AES encryption and MNIST inference.
- `ta/common/`: Shared TA utilities and model code.
- `proto/`: No‑std types and TA UUID shared by host and TA.
- `host/samples/`: Example inputs (images, raw digits, model file).
- Root `Makefile`: Orchestrates cross‑compiles for host and TA.

## Build, Test, and Development
- `make toolchain`: Install the Rust toolchain specified in `rust-toolchain.toml`.
- `make all`: Build host and TA with default features (`encrypt-model`).
- `make no-encrypt`: Production build excluding the `encrypt-model` feature.
- `make host` / `make ta`: Build components individually.
- Run host binary (after build): `host/target/<triple>/release/enc_mnist-rs ...`.
- Example flows:
  - Encrypt model: `./enc_mnist-rs encrypt-model --input ../samples/model.bin --output ./model_enc.json`
  - Inference (encrypted): `./enc_mnist-rs infer -m ./model_enc.json -i ../samples/7.png`

Notes:
- Cross settings come from `Makefile` (`TARGET_*`, `CROSS_COMPILE_*`). Ensure `TA_DEV_KIT_DIR` is set for signing.

## Coding Style & Conventions
- Rust 2021 edition; 4‑space indentation, no tabs.
- Prefer idiomatic Rust: `?` for error handling, small modules under `host/src/commands/`.
- Run format/lints before PRs: `cargo fmt --all` and `cargo clippy --all -- -D warnings` (per crate).
- Naming: use clear module scopes like `host/src/tee.rs`, `ta/inference/src/*`, `proto/src/*`.

## Testing Guidelines
- This example app uses sample artifacts in `host/samples/` for manual verification.
- If adding tests, place them next to the code (`mod tests`) and run `cargo test` inside each crate (`host`, `proto`, TA unit tests where applicable).
- Include short, reproducible CLI scripts for new features.

## Commit & Pull Requests
- Commits: imperative, scoped subjects, e.g. `host: add encrypted model CLI`, `ta: random IV generation`.
- PRs must include:
  - Summary, rationale, and affected paths.
  - Build steps used (`make all` or `make no-encrypt`).
  - Verification: sample commands and expected outputs/logs.
  - Linked issues and any security considerations.

## Security & Configuration
- Never log plaintext model data or keys on the host.
- Keep crypto in TA; host must only pass buffers.
- TA signing uses `ta/inference/uuid.txt` and `SIGN` script; set `TA_SIGN_KEY` if not using the default.
