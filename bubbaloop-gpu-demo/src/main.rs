use std::env;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use kornia::image::{ops, Image, ImageSize};
use kornia::imgproc;
use kornia::imgproc::interpolation::InterpolationMode;
use kornia::io::{functional as F, jpeg, png};
use kornia::io::v4l::{PixelFormat, V4LCameraConfig, V4lVideoCapture};
use kornia::tensor::CpuAllocator;

type DemoResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

const TOTAL_FRAMES: usize = 200;
const TARGET_SIZE: ImageSize = ImageSize {
    width: 640,
    height: 640,
};
const OUTPUT_PNG: &str = "output_frame.png";
const DEFAULT_FALLBACK_RELATIVE_PATH: &str = "assets/fallback.jpg";

struct Args {
    device: u32,
    fallback: Option<PathBuf>,
}

fn parse_args() -> DemoResult<Args> {
    let mut device = 0u32;
    let mut fallback = None;

    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--device" => {
                let value = it
                    .next()
                    .ok_or("missing value for --device")?
                    .parse::<u32>()
                    .map_err(|e| format!("invalid --device value: {e}"))?;
                device = value;
            }
            "--fallback" => {
                let value = it.next().ok_or("missing value for --fallback")?;
                fallback = Some(PathBuf::from(value));
            }
            _ => {
                return Err(format!("unknown argument: {arg}").into());
            }
        }
    }

    Ok(Args { device, fallback })
}

struct CameraSource {
    capture: V4lVideoCapture,
    rgb_buffer: Image<u8, 3, CpuAllocator>,
}

enum FrameMode {
    Camera(CameraSource),
    StaticImage(Image<u8, 3, CpuAllocator>),
}

struct FrameSource {
    mode: FrameMode,
    fallback: Option<Image<u8, 3, CpuAllocator>>,
}

impl FrameSource {
    fn from_fallback(path: &Path) -> DemoResult<Self> {
        let image = F::read_image_any_rgb8(path)?;
        println!("Input source: fallback image ({})", path.display());
        Ok(Self {
            mode: FrameMode::StaticImage(image),
            fallback: None,
        })
    }

    fn from_camera(device: u32) -> DemoResult<Self> {
        let fallback_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(DEFAULT_FALLBACK_RELATIVE_PATH);

        let fallback = if fallback_path.exists() {
            Some(F::read_image_any_rgb8(&fallback_path)?)
        } else {
            None
        };

        match Self::open_camera(device) {
            Ok(camera) => {
                println!("Input source: webcam (/dev/video{device})");
                Ok(Self {
                    mode: FrameMode::Camera(camera),
                    fallback,
                })
            }
            Err(err) => {
                if let Some(image) = fallback {
                    eprintln!(
                        "Webcam unavailable ({err}). Using fallback {}",
                        fallback_path.display()
                    );
                    Ok(Self {
                        mode: FrameMode::StaticImage(image),
                        fallback: None,
                    })
                } else {
                    Err(format!(
                        "Webcam unavailable ({err}) and fallback image not found at {}",
                        fallback_path.display()
                    )
                    .into())
                }
            }
        }
    }

    fn open_camera(device: u32) -> DemoResult<CameraSource> {
        let device_path = format!("/dev/video{device}");
        let candidates = [
            V4LCameraConfig {
                device_path: device_path.clone(),
                size: ImageSize {
                    width: 1920,
                    height: 1080,
                },
                format: PixelFormat::MJPG,
                fps: 30,
                ..Default::default()
            },
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

        let mut last_err = String::from("camera initialization failed");

        for config in candidates {
            let size = config.size;
            match V4lVideoCapture::new(config) {
                Ok(capture) => {
                    let rgb_buffer = Image::<u8, 3, _>::from_size_val(size, 0, CpuAllocator)?;
                    return Ok(CameraSource {
                        capture,
                        rgb_buffer,
                    });
                }
                Err(err) => {
                    last_err = err.to_string();
                }
            }
        }

        Err(last_err.into())
    }

    fn switch_to_fallback(&mut self, reason: &str) -> bool {
        if let Some(image) = self.fallback.take() {
            eprintln!("Switching to fallback image: {reason}");
            self.mode = FrameMode::StaticImage(image);
            true
        } else {
            false
        }
    }

    fn ensure_rgb_buffer(camera: &mut CameraSource, size: ImageSize) -> DemoResult<()> {
        if camera.rgb_buffer.size() != size {
            camera.rgb_buffer = Image::<u8, 3, _>::from_size_val(size, 0, CpuAllocator)?;
            eprintln!(
                "Reallocated webcam RGB buffer to {}x{}",
                size.width, size.height
            );
        }
        Ok(())
    }

    fn next_rgb8(&mut self) -> DemoResult<Option<&Image<u8, 3, CpuAllocator>>> {
        match &mut self.mode {
            FrameMode::StaticImage(image) => Ok(Some(image)),
            FrameMode::Camera(camera) => {
                let frame = match camera.capture.grab_frame() {
                    Ok(Some(frame)) => frame,
                    Ok(None) => return Ok(None),
                    Err(err) => return Err(format!("camera read failed: {err}").into()),
                };

                match frame.pixel_format {
                    PixelFormat::MJPG => {
                        let (jpeg_size, channels) = jpeg::decode_image_jpeg_info(frame.buffer.as_slice())?;
                        if channels != 3 {
                            return Err(format!("unsupported MJPG channel count: {channels}").into());
                        }
                        Self::ensure_rgb_buffer(camera, jpeg_size)?;
                        jpeg::decode_image_jpeg_rgb8(frame.buffer.as_slice(), &mut camera.rgb_buffer)?;
                    }
                    PixelFormat::YUYV => {
                        imgproc::color::convert_yuyv_to_rgb_u8(
                            frame.buffer.as_slice(),
                            &mut camera.rgb_buffer,
                            imgproc::color::YuvToRgbMode::Bt601Full,
                        )?;
                    }
                    other => return Err(format!("unsupported camera pixel format: {other}").into()),
                }

                Ok(Some(&camera.rgb_buffer))
            }
        }
    }
}

struct ImgprocPipeline {
    input_size: ImageSize,
    rgb_f32: Image<f32, 3, CpuAllocator>,
    gray_f32: Image<f32, 1, CpuAllocator>,
    resized_f32: Image<f32, 1, CpuAllocator>,
}

impl ImgprocPipeline {
    fn new(input_size: ImageSize) -> DemoResult<Self> {
        let rgb_f32 = Image::<f32, 3, _>::from_size_val(input_size, 0.0, CpuAllocator)?;
        let gray_f32 = Image::<f32, 1, _>::from_size_val(input_size, 0.0, CpuAllocator)?;
        let resized_f32 = Image::<f32, 1, _>::from_size_val(TARGET_SIZE, 0.0, CpuAllocator)?;

        Ok(Self {
            input_size,
            rgb_f32,
            gray_f32,
            resized_f32,
        })
    }

    fn run(&mut self, frame: &Image<u8, 3, CpuAllocator>) -> DemoResult<f64> {
        let start = Instant::now();
        ops::cast_and_scale(frame, &mut self.rgb_f32, 1.0 / 255.0)?;
        imgproc::color::gray_from_rgb(&self.rgb_f32, &mut self.gray_f32)?;
        imgproc::resize::resize_native(
            &self.gray_f32,
            &mut self.resized_f32,
            InterpolationMode::Bilinear,
        )?;
        Ok(start.elapsed().as_secs_f64() * 1000.0)
    }

    fn save_png(&self, path: &Path) -> DemoResult<()> {
        let output_u8 = self.resized_f32.scale_and_cast::<u8>(255.0)?;
        png::write_image_png_gray8(path, &output_u8)?;
        Ok(())
    }
}

#[derive(Default)]
struct Stats {
    count: usize,
    sum_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

impl Stats {
    fn update(&mut self, latency_ms: f64) {
        if self.count == 0 {
            self.min_ms = latency_ms;
            self.max_ms = latency_ms;
        } else {
            self.min_ms = self.min_ms.min(latency_ms);
            self.max_ms = self.max_ms.max(latency_ms);
        }

        self.count += 1;
        self.sum_ms += latency_ms;
    }

    fn avg_latency(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_ms / self.count as f64
        }
    }
}

fn main() -> DemoResult<()> {
    let args = parse_args()?;

    let mut source = if let Some(path) = args.fallback.as_ref() {
        FrameSource::from_fallback(path)?
    } else {
        FrameSource::from_camera(args.device)?
    };

    let mut pipeline: Option<ImgprocPipeline> = None;
    let mut stats = Stats::default();
    let wall_start = Instant::now();
    let mut frame_index = 0usize;

    while frame_index < TOTAL_FRAMES {
        let frame = match source.next_rgb8() {
            Ok(Some(frame)) => frame,
            Ok(None) => {
                thread::sleep(Duration::from_millis(2));
                continue;
            }
            Err(err) => {
                let reason = err.to_string();
                if source.switch_to_fallback(&reason) {
                    continue;
                }
                return Err(err);
            }
        };

        if pipeline
            .as_ref()
            .map(|p| p.input_size != frame.size())
            .unwrap_or(true)
        {
            pipeline = Some(ImgprocPipeline::new(frame.size())?);
        }

        let latency_ms = if let Some(proc_pipeline) = pipeline.as_mut() {
            proc_pipeline.run(frame)?
        } else {
            continue;
        };

        frame_index += 1;
        stats.update(latency_ms);

        let elapsed = wall_start.elapsed().as_secs_f64();
        let fps = if elapsed > 0.0 {
            frame_index as f64 / elapsed
        } else {
            0.0
        };

        println!(
            "Frame {:03} | imgproc: {:.2}ms | FPS: {:.1}",
            frame_index, latency_ms, fps
        );
    }

    let avg_fps = {
        let elapsed = wall_start.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            frame_index as f64 / elapsed
        } else {
            0.0
        }
    };

    if let Some(proc_pipeline) = pipeline.as_ref() {
        let out_path = PathBuf::from(OUTPUT_PNG);
        proc_pipeline.save_png(&out_path)?;
        println!("Saved output frame: {}", out_path.display());
    }

    println!("=== Summary ({TOTAL_FRAMES} frames) ===");
    println!(
        "Avg latency: {:.2}ms | Avg FPS: {:.2} | Min: {:.2}ms | Max: {:.2}ms",
        stats.avg_latency(),
        avg_fps,
        stats.min_ms,
        stats.max_ms
    );

    Ok(())
}
