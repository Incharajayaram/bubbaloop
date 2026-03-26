use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bubbaloop_node_sdk::anyhow::{anyhow, Result};
use bubbaloop_node_sdk::async_trait::async_trait;
use bubbaloop_node_sdk::log;
use bubbaloop_node_sdk::zenoh;
use bubbaloop_node_sdk::{Node, NodeContext};
use bubbaloop_schemas::Header;
use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};
use kornia::image::allocator::CpuAllocator;
use kornia::image::{Image, ImageSize};
use kornia::imgproc;
use kornia::io::jpeg;
use kornia::io::v4l::{PixelFormat, V4LCameraConfig, V4lVideoCapture};
use kornia_vlm::smolvlm2::{InputMedia, Line, Message as VlmMessage, Role, SmolVlm2, SmolVlm2Config};
use prost::Message;
use tokio::sync::mpsc;

pub mod camera {
    pub mod v1 {
        include!(concat!(env!("OUT_DIR"), "/bubbaloop.camera.v1.rs"));
    }
}

use camera::v1::CompressedImage;

const DESCRIPTOR: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/descriptor.bin"));

#[cube(launch_unchecked)]
fn gray_from_rgb_kernel(rgb: &Array<f32>, gray: &mut Array<f32>) {
    if ABSOLUTE_POS < gray.len() {
        let base = ABSOLUTE_POS * 3;
        let r = rgb[base];
        let g = rgb[base + 1];
        let b = rgb[base + 2];
        gray[ABSOLUTE_POS] = 0.299 * r + 0.587 * g + 0.114 * b;
    }
}

#[cube(launch_unchecked)]
fn resize_bilinear_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    #[comptime] src_w: usize,
    #[comptime] src_h: usize,
    #[comptime] dst_w: usize,
    #[comptime] dst_h: usize,
) {
    let out_idx = ABSOLUTE_POS;
    if out_idx < dst_w * dst_h {
        let x_out = out_idx % dst_w;
        let y_out = out_idx / dst_w;

        let scale_x = (src_w - 1) as f32 / dst_w as f32;
        let scale_y = (src_h - 1) as f32 / dst_h as f32;
        let x_src_f = x_out as f32 * scale_x;
        let y_src_f = y_out as f32 * scale_y;

        let x0 = x_src_f as usize;
        let y0 = y_src_f as usize;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let wx = x_src_f - x0 as f32;
        let wy = y_src_f - y0 as f32;

        let p00 = src[y0 * src_w + x0];
        let p01 = src[y0 * src_w + x1];
        let p10 = src[y1 * src_w + x0];
        let p11 = src[y1 * src_w + x1];

        dst[out_idx] = p00 * (1.0 - wx) * (1.0 - wy)
            + p01 * wx * (1.0 - wy)
            + p10 * (1.0 - wx) * wy
            + p11 * wx * wy;
    }
}

type RgbImage = Image<u8, 3, CpuAllocator>;
type GpuClient = ComputeClient<WgpuRuntime>;

struct FramePacket {
    frame: RgbImage,
    acq_time: u64,
}

struct CameraSource {
    capture: V4lVideoCapture,
    rgb_buffer: RgbImage,
}

fn now_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_nanos() as u64
}

fn launch_shape_1d(n: usize) -> (CubeCount, CubeDim) {
    let cube_dim_x = 256u32;
    let cubes_x = (n as u32).div_ceil(cube_dim_x);
    (CubeCount::new_1d(cubes_x), CubeDim::new_1d(cube_dim_x))
}

fn sync_client(client: &GpuClient) -> Result<()> {
    pollster::block_on(client.sync())?;
    Ok(())
}

fn open_camera(device_id: u32) -> Result<CameraSource> {
    let device_path = format!("/dev/video{device_id}");
    let candidates = [
        V4LCameraConfig {
            device_path: device_path.clone(),
            size: ImageSize {
                width: 1280,
                height: 720,
            },
            format: PixelFormat::MJPG,
            fps: 30,
            ..Default::default()
        },
        V4LCameraConfig {
            device_path: device_path.clone(),
            size: ImageSize {
                width: 640,
                height: 480,
            },
            format: PixelFormat::MJPG,
            fps: 30,
            ..Default::default()
        },
        V4LCameraConfig {
            device_path,
            size: ImageSize {
                width: 640,
                height: 480,
            },
            format: PixelFormat::YUYV,
            fps: 30,
            ..Default::default()
        },
    ];

    let mut last_error = String::from("camera init failed");

    for cfg in candidates {
        let size = cfg.size;
        match V4lVideoCapture::new(cfg) {
            Ok(capture) => {
                let rgb_buffer = Image::<u8, 3, _>::from_size_val(size, 0, CpuAllocator)?;
                return Ok(CameraSource {
                    capture,
                    rgb_buffer,
                });
            }
            Err(err) => {
                last_error = err.to_string();
            }
        }
    }

    Err(anyhow!(last_error))
}

fn ensure_rgb_buffer(camera: &mut CameraSource, size: ImageSize) -> Result<()> {
    if camera.rgb_buffer.size() != size {
        camera.rgb_buffer = Image::<u8, 3, _>::from_size_val(size, 0, CpuAllocator)?;
    }
    Ok(())
}

fn spawn_capture_thread(device_id: u32, frame_tx: mpsc::Sender<FramePacket>) {
    thread::spawn(move || {
        let mut camera = match open_camera(device_id) {
            Ok(cam) => cam,
            Err(err) => {
                log::error!("camera open failed: {err}");
                return;
            }
        };

        log::info!("camera capture started on /dev/video{device_id}");

        loop {
            let frame = match camera.capture.grab_frame() {
                Ok(Some(frame)) => frame,
                Ok(None) => {
                    thread::sleep(Duration::from_millis(2));
                    continue;
                }
                Err(err) => {
                    log::warn!("camera read failed: {err}");
                    thread::sleep(Duration::from_millis(20));
                    continue;
                }
            };

            let decode_result: Result<()> = match frame.pixel_format {
                PixelFormat::MJPG => {
                    let layout = match jpeg::decode_image_jpeg_layout(frame.buffer.as_slice()) {
                        Ok(info) => info,
                        Err(err) => {
                            log::warn!("jpeg layout decode failed: {err}");
                            continue;
                        }
                    };

                    if layout.channels != 3 {
                        log::warn!("unsupported MJPG channels: {}", layout.channels);
                        continue;
                    }

                    if let Err(err) = ensure_rgb_buffer(&mut camera, layout.image_size) {
                        log::warn!("rgb buffer resize failed: {err}");
                        continue;
                    }

                    jpeg::decode_image_jpeg_rgb8(frame.buffer.as_slice(), &mut camera.rgb_buffer)
                        .map_err(|e| anyhow!(e.to_string()))
                }
                PixelFormat::YUYV => imgproc::color::convert_yuyv_to_rgb_u8(
                    frame.buffer.as_slice(),
                    &mut camera.rgb_buffer,
                    imgproc::color::YuvToRgbMode::Bt601Full,
                )
                .map_err(|e| anyhow!(e.to_string())),
                other => {
                    log::warn!("unsupported pixel format: {other}");
                    continue;
                }
            };

            if let Err(err) = decode_result {
                log::warn!("frame decode failed: {err}");
                continue;
            }

            let packet = FramePacket {
                frame: camera.rgb_buffer.clone(),
                acq_time: now_nanos(),
            };

            if frame_tx.blocking_send(packet).is_err() {
                break;
            }
        }

        log::info!("camera capture thread stopped");
    });
}

struct GpuPipeline {
    client: GpuClient,
    dst_w: usize,
    dst_h: usize,
}

impl GpuPipeline {
    fn new(dst_w: usize, dst_h: usize) -> Self {
        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);
        Self { client, dst_w, dst_h }
    }

    fn process(&self, frame: &RgbImage) -> Result<Vec<f32>> {
        let src_w = frame.width();
        let src_h = frame.height();
        let n_rgb = src_w * src_h * 3;
        let n_gray = src_w * src_h;
        let n_out = self.dst_w * self.dst_h;

        let rgb_f32: Vec<f32> = frame
            .as_slice()
            .iter()
            .map(|&v| v as f32 / 255.0)
            .collect();

        if rgb_f32.len() != n_rgb {
            return Err(anyhow!("unexpected rgb buffer size"));
        }

        let rgb_handle = self.client.create_from_slice(f32::as_bytes(&rgb_f32));
        let gray_handle = self.client.empty(n_gray * std::mem::size_of::<f32>());
        let dst_handle = self.client.empty(n_out * std::mem::size_of::<f32>());

        let (gray_count, gray_dim) = launch_shape_1d(n_gray);
        unsafe {
            gray_from_rgb_kernel::launch_unchecked::<WgpuRuntime>(
                &self.client,
                gray_count,
                gray_dim,
                ArrayArg::from_raw_parts::<f32>(&rgb_handle, n_rgb, 1),
                ArrayArg::from_raw_parts::<f32>(&gray_handle, n_gray, 1),
            )?;
        }

        let (resize_count, resize_dim) = launch_shape_1d(n_out);
        unsafe {
            resize_bilinear_kernel::launch_unchecked::<WgpuRuntime>(
                &self.client,
                resize_count,
                resize_dim,
                ArrayArg::from_raw_parts::<f32>(&gray_handle, n_gray, 1),
                ArrayArg::from_raw_parts::<f32>(&dst_handle, n_out, 1),
                src_w,
                src_h,
                self.dst_w,
                self.dst_h,
            )?;
        }

        sync_client(&self.client)?;
        let out = f32::from_bytes(&self.client.read_one(dst_handle)).to_vec();
        Ok(out)
    }
}

fn encode_jpeg_gray(data: &[f32], width: u32, height: u32) -> Result<Vec<u8>> {
    let gray_u8: Vec<u8> = data
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
        .collect();

    let img = image::GrayImage::from_raw(width, height, gray_u8)
        .ok_or_else(|| anyhow!("failed to create GrayImage"))?;

    let mut bytes = Vec::new();
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 85);
    encoder.encode_image(&img)?;
    Ok(bytes)
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub device_id: u32,
    pub resize_width: u32,
    pub resize_height: u32,
    pub rate_hz: f64,
}

pub struct KorniaGpuNode {
    publisher: zenoh::pubsub::Publisher<'static>,
    vlm_publisher: zenoh::pubsub::Publisher<'static>,
    frame_rx: mpsc::Receiver<FramePacket>,
    gpu: Arc<GpuPipeline>,
    vlm: SmolVlm2<32, CpuAllocator>,
    resize_width: u32,
    resize_height: u32,
    scope: String,
    machine_id: String,
    publish_period: Option<Duration>,
}

#[async_trait]
impl Node for KorniaGpuNode {
    type Config = Config;

    fn name() -> &'static str {
        "kornia-gpu-node"
    }

    fn descriptor() -> &'static [u8] {
        DESCRIPTOR
    }

    async fn init(ctx: &NodeContext, config: &Config) -> Result<Self> {
        let topic = ctx.topic("camera/gpu-processed/compressed");
        let publisher = ctx
            .session
            .declare_publisher(topic.clone())
            .await
            .map_err(|e| anyhow!("declare_publisher failed: {e}"))?;

        let vlm_topic = ctx.topic("camera/gpu-processed/vlm-label");
        let vlm_publisher = ctx
            .session
            .declare_publisher(vlm_topic.clone())
            .await
            .map_err(|e| anyhow!("declare_publisher vlm failed: {e}"))?;

        log::info!("loading SmolVLM2 model (this can take time on first run)...");
        let vlm_config = SmolVlm2Config {
            do_sample: false,
            debug: false,
            ..Default::default()
        };
        let vlm = SmolVlm2::<32, CpuAllocator>::new(vlm_config)
            .map_err(|e| anyhow!("SmolVLM2 init failed: {e}"))?;

        let (frame_tx, frame_rx) = mpsc::channel::<FramePacket>(4);
        spawn_capture_thread(config.device_id, frame_tx);

        let gpu = Arc::new(GpuPipeline::new(
            config.resize_width as usize,
            config.resize_height as usize,
        ));

        let publish_period = if config.rate_hz > 0.0 {
            Some(Duration::from_secs_f64(1.0 / config.rate_hz))
        } else {
            None
        };

        log::info!("publishing to topic: {}", topic);
        log::info!("publishing VLM labels to topic: {}", vlm_topic);

        Ok(Self {
            publisher,
            vlm_publisher,
            frame_rx,
            gpu,
            vlm,
            resize_width: config.resize_width,
            resize_height: config.resize_height,
            scope: ctx.scope.clone(),
            machine_id: ctx.machine_id.clone(),
            publish_period,
        })
    }

    async fn run(mut self, ctx: NodeContext) -> Result<()> {
        let mut shutdown_rx = ctx.shutdown_rx.clone();
        let mut sequence: u32 = 0;

        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    break;
                }
                maybe_packet = self.frame_rx.recv() => {
                    let packet = match maybe_packet {
                        Some(packet) => packet,
                        None => {
                            return Err(anyhow!("camera stream ended"));
                        }
                    };

                    log::info!("received frame seq={} size={}x{}", sequence, packet.frame.width(), packet.frame.height());
                    let processed = match self.gpu.process(&packet.frame) {
                        Ok(v) => v,
                        Err(err) => {
                            log::warn!("gpu process failed: {err}");
                            continue;
                        }
                    };

                    log::info!("gpu done seq={} gray_len={}", sequence, processed.len());
                    let jpeg = match encode_jpeg_gray(&processed, self.resize_width, self.resize_height) {
                        Ok(v) => v,
                        Err(err) => {
                            log::warn!("jpeg encode failed: {err}");
                            continue;
                        }
                    };

                    if sequence % 30 == 0 {
                        if let Err(e) = self.vlm.clear_context() {
                            log::warn!("vlm clear_context failed: {e}");
                        }

                        let response = self.vlm.inference(
                            vec![VlmMessage {
                                role: Role::User,
                                content: vec![
                                    Line::Image,
                                    Line::Text {
                                        text: "Briefly describe what you see in one sentence.".to_string(),
                                    },
                                ],
                            }],
                            Some(InputMedia::Images(vec![packet.frame.clone()])),
                            80,
                            CpuAllocator,
                        );

                        match response {
                            Ok(label) => {
                                log::info!("vlm seq={} label={}", sequence, label);
                                if let Err(e) = self.vlm_publisher.put(label.into_bytes()).await {
                                    log::warn!("vlm label publish failed seq={}: {}", sequence, e);
                                }
                            }
                            Err(e) => log::warn!("vlm inference failed seq={}: {}", sequence, e),
                        }
                    }

                    let pub_time = now_nanos();
                    let msg = CompressedImage {
                        header: Some(Header {
                            acq_time: packet.acq_time,
                            pub_time,
                            sequence,
                            frame_id: "gpu-webcam".to_string(),
                            machine_id: self.machine_id.clone(),
                            scope: self.scope.clone(),
                        }),
                        format: "jpeg".to_string(),
                        data: jpeg,
                        width: self.resize_width,
                        height: self.resize_height,
                    };

                    let payload = msg.encode_to_vec();
                    log::info!("publishing seq={} jpeg_bytes={}", sequence, msg.data.len());
                    if let Err(err) = self.publisher.put(payload).await {
                        log::warn!("publish failed: {err}");
                        continue;
                    }

                    sequence = sequence.wrapping_add(1);

                    if let Some(period) = self.publish_period {
                        tokio::time::sleep(period).await;
                    }
                }
            }
        }

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    bubbaloop_node_sdk::run_node::<KorniaGpuNode>().await
}
