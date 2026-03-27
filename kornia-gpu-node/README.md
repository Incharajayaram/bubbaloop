# Kornia GPU Node Demo

`kornia-gpu-node` is a Bubbaloop node that:

1. captures webcam frames from V4L2
2. preprocesses them with `kornia-rs`
3. publishes a JPEG-compressed stream to the Bubbaloop dashboard
4. optionally runs `SmolVLM2` caption generation when built with the `vlm` feature

This repo was used for the end-to-end demo portion of the project.

## Demo Modes

There are three useful modes:

1. Local GPU preprocess demo
2. Local CPU baseline demo
3. Cloud VLM-enabled build

## Repository Layout Assumption

This node expects the following layout:

```text
~/kornia-rs
~/bubbaloop
```

The `Cargo.toml` in this crate references `../../kornia-rs/...` paths accordingly.

## Prerequisites

Local machine:

- webcam at `/dev/video0` or another V4L2 device
- Rust toolchain
- local Bubbaloop checkout
- working dashboard setup
- `protobuf-compiler`
- `clang` and `libclang-dev` for the V4L bindings

Cloud GPU machine:

- Rust toolchain
- CUDA 12.5
- `flash-attn` capable GPU
- no webcam is required for the VLM-only validation

Ubuntu packages we ended up needing locally:

```bash
sudo apt-get update
sudo apt-get install -y protobuf-compiler clang libclang-dev
```

## Local Bubbaloop Setup

Start the router and bridge:

```bash
cd ~/bubbaloop
pixi run up
```

Start the dashboard:

```bash
cd ~/bubbaloop
pixi run dashboard
```

Then open:

```text
https://localhost:5173
```

If your local stack uses the production dashboard service instead, use the URL printed by
`pixi run dashboard`.

## Config

The node is controlled by `config.yaml`.

### GPU preprocess mode

```yaml
device_id: 0
resize_width: 640
resize_height: 640
rate_hz: 15.0
use_gpu: true
enable_vlm: false
vlm_every_n_frames: 30
vlm_prompt: "Briefly describe what you see in one sentence."
```

### CPU baseline mode

```yaml
device_id: 0
resize_width: 640
resize_height: 640
rate_hz: 15.0
use_gpu: false
enable_vlm: false
vlm_every_n_frames: 30
vlm_prompt: "Briefly describe what you see in one sentence."
```

### VLM-enabled mode

```yaml
device_id: 0
resize_width: 640
resize_height: 640
rate_hz: 15.0
use_gpu: true
enable_vlm: true
vlm_every_n_frames: 30
vlm_prompt: "Briefly describe what you see in one sentence."
```

## Build

Local preprocess-only build:

```bash
cd ~/bubbaloop/kornia-gpu-node
cargo build --release
```

VLM-enabled build:

```bash
cd ~/bubbaloop/kornia-gpu-node
cargo build --release --features vlm
```

## Run

Local GPU preprocess demo:

```bash
cd ~/bubbaloop/kornia-gpu-node
RUST_LOG=info cargo run --release -- -c config.yaml
```

Local VLM-enabled run:

```bash
cd ~/bubbaloop/kornia-gpu-node
RUST_LOG=info cargo run --release --features vlm -- -c config.yaml
```

For the final demo we used this split:

1. local laptop for the live webcam and dashboard demo
2. cloud GPU for `SmolVLM2` validation

That avoided forcing webcam capture through the cloud VM while still proving the VLM path on a
strong enough GPU.

## What To Look For In The Logs

The node logs per-frame timings like:

```text
seq=123 mode=gpu preprocess=13.30ms jpeg=13.17ms total=27.64ms bytes=...
```

When VLM is enabled, the log also includes:

```text
vlm seq=... label=...
seq=... mode=gpu preprocess=... jpeg=... vlm=... total=... bytes=...
```

These fields were used to measure:

- local CPU total pipeline latency
- local GPU total pipeline latency
- local preprocess and JPEG stage timing

Reference numbers from the recorded demo runs:

- CPU total pipeline: `23.24ms` (`43.04 FPS`)
- GPU total pipeline: `27.64ms` (`36.18 FPS`)
- GPU preprocess stage: `13.30ms`
- GPU JPEG stage: `13.17ms`

## Benchmark Collection

### CPU total pipeline

Run the node in CPU mode and save logs:

```bash
cd ~/bubbaloop/kornia-gpu-node
RUST_LOG=info cargo run --release -- -c config.yaml 2>&1 | tee /tmp/kornia_cpu_node.log
```

After letting it run for a short while, compute the average:

```bash
grep -o 'total=[0-9]\+\.[0-9]\+ms' /tmp/kornia_cpu_node.log | sed 's/.*=//;s/ms//' | tail -n 50 | awk '{sum+=$1; n++} END {if(n) {avg=sum/n; printf "cpu_avg_total_ms=%.2f over %d frames | fps=%.2f\n", avg, n, 1000/avg}}'
```

### GPU total pipeline

Run the node in GPU mode and save logs:

```bash
cd ~/bubbaloop/kornia-gpu-node
RUST_LOG=info cargo run --release -- -c config.yaml 2>&1 | tee /tmp/kornia_gpu_node.log
```

Average total time:

```bash
grep -o 'total=[0-9]\+\.[0-9]\+ms' /tmp/kornia_gpu_node.log | sed 's/.*=//;s/ms//' | tail -n 50 | awk '{sum+=$1; n++} END {if(n) {avg=sum/n; printf "gpu_avg_total_ms=%.2f over %d frames | fps=%.2f\n", avg, n, 1000/avg}}'
```

Average preprocess stage:

```bash
grep -o 'preprocess=[0-9]\+\.[0-9]\+ms' /tmp/kornia_gpu_node.log | sed 's/.*=//;s/ms//' | tail -n 50 | awk '{sum+=$1; n++} END {if(n) printf "gpu_preprocess_ms=%.2f over %d frames\n", sum/n, n}'
```

Average JPEG stage:

```bash
grep -o 'jpeg=[0-9]\+\.[0-9]\+ms' /tmp/kornia_gpu_node.log | sed 's/.*=//;s/ms//' | tail -n 50 | awk '{sum+=$1; n++} END {if(n) printf "gpu_jpeg_ms=%.2f over %d frames\n", sum/n, n}'
```

## Cloud GPU Notes

The full `SmolVLM2` validation was run on a stronger cloud GPU rather than on the local laptop GPU.

Use the `kornia-rs` side for that:

- `../../kornia-rs/examples/kornia-gpu-benchmark/README.md`

In practice, the final demo used:

- local machine for the live Bubbaloop webcam/dashboard pipeline
- cloud GPU for `SmolVLM2` inference validation

## Troubleshooting

If the node cannot connect to Zenoh:

```text
Failed to open Zenoh session: Unable to connect to any of [tcp/127.0.0.1:7447]
```

make sure the router is running locally:

```bash
cd ~/bubbaloop
pixi run up
```

If the dashboard shows `Waiting for keyframe...`, verify:

- the bridge is running
- the node is publishing to `camera/gpu-processed/compressed`
- the dashboard is connected to the local bridge
