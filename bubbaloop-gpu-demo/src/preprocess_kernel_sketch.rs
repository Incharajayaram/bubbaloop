//! GPU preprocessing dispatch sketch.
//!
//! This file keeps `warp_perspective` as one representative geometric kernel,
//! but expands the sketch to the preprocessing stages that are more directly
//! tied to the Bubbaloop camera pipeline:
//! - `gray_from_rgb`
//! - `resize_bilinear`
//! - `normalize + HWC -> CHW`
//! - a fused preprocessing path
//! - `warp_perspective` as a nontrivial geometric transform
//!
//! The intent is to show how allocator-based dispatch can scale from the
//! current CPU implementation to a future GPU backend without implicit copies.

pub struct CpuAllocator;
pub struct GpuAllocator {
    pub device_id: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageSize {
    pub width: usize,
    pub height: usize,
}

pub struct Image<T, const C: usize, A> {
    _marker: std::marker::PhantomData<(T, A)>,
}

pub struct Tensor<T, A> {
    _marker: std::marker::PhantomData<(T, A)>,
}

#[derive(Debug)]
pub struct ImageError(pub String);

impl std::fmt::Display for ImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ImageError: {}", self.0)
    }
}

pub trait ImageAllocator {}
impl ImageAllocator for CpuAllocator {}
impl ImageAllocator for GpuAllocator {}

pub trait CpuAccessible {}
impl CpuAccessible for CpuAllocator {}

#[derive(Clone, Copy, Debug)]
pub enum InterpolationMode {
    Nearest,
    Bilinear,
}

#[derive(Clone, Copy, Debug)]
pub struct NormalizeConfig<const C: usize> {
    pub mean: [f32; C],
    pub std: [f32; C],
}

#[derive(Clone, Copy, Debug)]
pub struct PreprocessConfig<const C: usize> {
    pub out_size: ImageSize,
    pub interpolation: InterpolationMode,
    pub normalize: NormalizeConfig<C>,
    pub grayscale: bool,
}

pub trait GrayFromRgbKernel<A: ImageAllocator> {
    fn gray_from_rgb_f32(
        src: &Image<f32, 3, A>,
        dst: &mut Image<f32, 1, A>,
    ) -> Result<(), ImageError>
    where
        Self: Sized;
}

pub trait ResizeKernel<const C: usize, A: ImageAllocator> {
    fn resize_f32(
        src: &Image<f32, C, A>,
        dst: &mut Image<f32, C, A>,
        mode: InterpolationMode,
    ) -> Result<(), ImageError>
    where
        Self: Sized;
}

pub trait NormalizeKernel<const C: usize, A: ImageAllocator> {
    fn normalize_hwc_to_chw_f32(
        src: &Image<f32, C, A>,
        dst: &mut Tensor<f32, A>,
        config: NormalizeConfig<C>,
    ) -> Result<(), ImageError>
    where
        Self: Sized;
}

pub trait WarpPerspectiveKernel<const C: usize, A: ImageAllocator> {
    fn warp_perspective_f32(
        src: &Image<f32, C, A>,
        dst: &mut Image<f32, C, A>,
        m: &[f32; 9],
    ) -> Result<(), ImageError>
    where
        Self: Sized;
}

pub trait FusedPreprocessKernel<A: ImageAllocator> {
    fn preprocess_rgb8_to_chw_f32(
        src: &Image<u8, 3, A>,
        dst: &mut Tensor<f32, A>,
        out_size: ImageSize,
        normalize: NormalizeConfig<3>,
    ) -> Result<(), ImageError>
    where
        Self: Sized;
}

pub trait VisionPreprocessPipeline<A: ImageAllocator> {
    fn camera_frame_to_model_tensor(
        src: &Image<u8, 3, A>,
        dst: &mut Tensor<f32, A>,
        config: PreprocessConfig<3>,
    ) -> Result<(), ImageError>
    where
        Self: Sized;
}

pub struct CpuPreprocessKernels;

impl GrayFromRgbKernel<CpuAllocator> for CpuPreprocessKernels {
    fn gray_from_rgb_f32(
        _src: &Image<f32, 3, CpuAllocator>,
        _dst: &mut Image<f32, 1, CpuAllocator>,
    ) -> Result<(), ImageError> {
        Ok(())
    }
}

impl<const C: usize> ResizeKernel<C, CpuAllocator> for CpuPreprocessKernels {
    fn resize_f32(
        _src: &Image<f32, C, CpuAllocator>,
        _dst: &mut Image<f32, C, CpuAllocator>,
        _mode: InterpolationMode,
    ) -> Result<(), ImageError> {
        Ok(())
    }
}

impl<const C: usize> NormalizeKernel<C, CpuAllocator> for CpuPreprocessKernels {
    fn normalize_hwc_to_chw_f32(
        _src: &Image<f32, C, CpuAllocator>,
        _dst: &mut Tensor<f32, CpuAllocator>,
        _config: NormalizeConfig<C>,
    ) -> Result<(), ImageError> {
        Ok(())
    }
}

impl<const C: usize> WarpPerspectiveKernel<C, CpuAllocator> for CpuPreprocessKernels {
    fn warp_perspective_f32(
        _src: &Image<f32, C, CpuAllocator>,
        _dst: &mut Image<f32, C, CpuAllocator>,
        _m: &[f32; 9],
    ) -> Result<(), ImageError> {
        Ok(())
    }
}

impl FusedPreprocessKernel<CpuAllocator> for CpuPreprocessKernels {
    fn preprocess_rgb8_to_chw_f32(
        _src: &Image<u8, 3, CpuAllocator>,
        _dst: &mut Tensor<f32, CpuAllocator>,
        _out_size: ImageSize,
        _normalize: NormalizeConfig<3>,
    ) -> Result<(), ImageError> {
        Ok(())
    }
}

impl VisionPreprocessPipeline<CpuAllocator> for CpuPreprocessKernels {
    fn camera_frame_to_model_tensor(
        src: &Image<u8, 3, CpuAllocator>,
        dst: &mut Tensor<f32, CpuAllocator>,
        config: PreprocessConfig<3>,
    ) -> Result<(), ImageError> {
        let _ = config.grayscale;
        Self::preprocess_rgb8_to_chw_f32(src, dst, config.out_size, config.normalize)
    }
}

pub struct GpuPreprocessKernels;

impl GrayFromRgbKernel<GpuAllocator> for GpuPreprocessKernels {
    fn gray_from_rgb_f32(
        _src: &Image<f32, 3, GpuAllocator>,
        _dst: &mut Image<f32, 1, GpuAllocator>,
    ) -> Result<(), ImageError> {
        Ok(())
    }
}

impl<const C: usize> ResizeKernel<C, GpuAllocator> for GpuPreprocessKernels {
    fn resize_f32(
        _src: &Image<f32, C, GpuAllocator>,
        _dst: &mut Image<f32, C, GpuAllocator>,
        _mode: InterpolationMode,
    ) -> Result<(), ImageError> {
        Ok(())
    }
}

impl<const C: usize> NormalizeKernel<C, GpuAllocator> for GpuPreprocessKernels {
    fn normalize_hwc_to_chw_f32(
        _src: &Image<f32, C, GpuAllocator>,
        _dst: &mut Tensor<f32, GpuAllocator>,
        _config: NormalizeConfig<C>,
    ) -> Result<(), ImageError> {
        Ok(())
    }
}

impl<const C: usize> WarpPerspectiveKernel<C, GpuAllocator> for GpuPreprocessKernels {
    fn warp_perspective_f32(
        _src: &Image<f32, C, GpuAllocator>,
        _dst: &mut Image<f32, C, GpuAllocator>,
        _m: &[f32; 9],
    ) -> Result<(), ImageError> {
        Ok(())
    }
}

impl FusedPreprocessKernel<GpuAllocator> for GpuPreprocessKernels {
    fn preprocess_rgb8_to_chw_f32(
        _src: &Image<u8, 3, GpuAllocator>,
        _dst: &mut Tensor<f32, GpuAllocator>,
        _out_size: ImageSize,
        _normalize: NormalizeConfig<3>,
    ) -> Result<(), ImageError> {
        Ok(())
    }
}

impl VisionPreprocessPipeline<GpuAllocator> for GpuPreprocessKernels {
    fn camera_frame_to_model_tensor(
        src: &Image<u8, 3, GpuAllocator>,
        dst: &mut Tensor<f32, GpuAllocator>,
        config: PreprocessConfig<3>,
    ) -> Result<(), ImageError> {
        let _ = config.interpolation;
        let _ = config.grayscale;
        Self::preprocess_rgb8_to_chw_f32(src, dst, config.out_size, config.normalize)
    }
}

impl<T, const C: usize> Image<T, C, CpuAllocator> {
    pub fn to_device(&self, _alloc: GpuAllocator) -> Result<Image<T, C, GpuAllocator>, ImageError> {
        Ok(Image {
            _marker: std::marker::PhantomData,
        })
    }
}

impl<T, const C: usize> Image<T, C, GpuAllocator> {
    pub fn to_host(&self, _alloc: CpuAllocator) -> Result<Image<T, C, CpuAllocator>, ImageError> {
        Ok(Image {
            _marker: std::marker::PhantomData,
        })
    }
}

impl<T> Tensor<T, CpuAllocator> {
    pub fn to_device(&self, _alloc: GpuAllocator) -> Result<Tensor<T, GpuAllocator>, ImageError> {
        Ok(Tensor {
            _marker: std::marker::PhantomData,
        })
    }
}

impl<T> Tensor<T, GpuAllocator> {
    pub fn to_host(&self, _alloc: CpuAllocator) -> Result<Tensor<T, CpuAllocator>, ImageError> {
        Ok(Tensor {
            _marker: std::marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_preprocess_kernels_compile() {
        let rgb: Image<f32, 3, CpuAllocator> = Image {
            _marker: std::marker::PhantomData,
        };
        let mut gray: Image<f32, 1, CpuAllocator> = Image {
            _marker: std::marker::PhantomData,
        };
        let mut resized: Image<f32, 1, CpuAllocator> = Image {
            _marker: std::marker::PhantomData,
        };
        let mut chw: Tensor<f32, CpuAllocator> = Tensor {
            _marker: std::marker::PhantomData,
        };
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0f32];

        CpuPreprocessKernels::gray_from_rgb_f32(&rgb, &mut gray).unwrap();
        CpuPreprocessKernels::resize_f32(&gray, &mut resized, InterpolationMode::Bilinear)
            .unwrap();
        CpuPreprocessKernels::normalize_hwc_to_chw_f32(
            &rgb,
            &mut chw,
            NormalizeConfig {
                mean: [0.5, 0.5, 0.5],
                std: [0.5, 0.5, 0.5],
            },
        )
        .unwrap();
        CpuPreprocessKernels::warp_perspective_f32(&rgb, &mut Image {
            _marker: std::marker::PhantomData,
        }, &identity)
        .unwrap();
    }

    #[test]
    fn gpu_preprocess_kernels_compile() {
        let rgb: Image<f32, 3, GpuAllocator> = Image {
            _marker: std::marker::PhantomData,
        };
        let rgb_u8: Image<u8, 3, GpuAllocator> = Image {
            _marker: std::marker::PhantomData,
        };
        let mut gray: Image<f32, 1, GpuAllocator> = Image {
            _marker: std::marker::PhantomData,
        };
        let mut resized: Image<f32, 1, GpuAllocator> = Image {
            _marker: std::marker::PhantomData,
        };
        let mut chw: Tensor<f32, GpuAllocator> = Tensor {
            _marker: std::marker::PhantomData,
        };
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0f32];

        GpuPreprocessKernels::gray_from_rgb_f32(&rgb, &mut gray).unwrap();
        GpuPreprocessKernels::resize_f32(&gray, &mut resized, InterpolationMode::Bilinear)
            .unwrap();
        GpuPreprocessKernels::normalize_hwc_to_chw_f32(
            &rgb,
            &mut chw,
            NormalizeConfig {
                mean: [0.485, 0.456, 0.406],
                std: [0.229, 0.224, 0.225],
            },
        )
        .unwrap();
        GpuPreprocessKernels::preprocess_rgb8_to_chw_f32(
            &rgb_u8,
            &mut chw,
            ImageSize {
                width: 640,
                height: 640,
            },
            NormalizeConfig {
                mean: [0.485, 0.456, 0.406],
                std: [0.229, 0.224, 0.225],
            },
        )
        .unwrap();
        GpuPreprocessKernels::warp_perspective_f32(&rgb, &mut Image {
            _marker: std::marker::PhantomData,
        }, &identity)
        .unwrap();
    }

    #[test]
    fn explicit_transfer_compiles() {
        let cpu_img: Image<f32, 3, CpuAllocator> = Image {
            _marker: std::marker::PhantomData,
        };
        let gpu_img = cpu_img.to_device(GpuAllocator { device_id: 0 }).unwrap();
        let _back = gpu_img.to_host(CpuAllocator).unwrap();

        let cpu_tensor: Tensor<f32, CpuAllocator> = Tensor {
            _marker: std::marker::PhantomData,
        };
        let gpu_tensor = cpu_tensor.to_device(GpuAllocator { device_id: 0 }).unwrap();
        let _tensor_back = gpu_tensor.to_host(CpuAllocator).unwrap();
    }

    #[test]
    fn end_to_end_pipeline_shape_compiles() {
        let cpu_rgb_u8: Image<u8, 3, CpuAllocator> = Image {
            _marker: std::marker::PhantomData,
        };
        let mut cpu_tensor: Tensor<f32, CpuAllocator> = Tensor {
            _marker: std::marker::PhantomData,
        };
        let config = PreprocessConfig {
            out_size: ImageSize {
                width: 640,
                height: 640,
            },
            interpolation: InterpolationMode::Bilinear,
            normalize: NormalizeConfig {
                mean: [0.485, 0.456, 0.406],
                std: [0.229, 0.224, 0.225],
            },
            grayscale: false,
        };

        CpuPreprocessKernels::camera_frame_to_model_tensor(&cpu_rgb_u8, &mut cpu_tensor, config)
            .unwrap();

        let gpu_rgb_u8 = cpu_rgb_u8.to_device(GpuAllocator { device_id: 0 }).unwrap();
        let mut gpu_tensor = cpu_tensor.to_device(GpuAllocator { device_id: 0 }).unwrap();

        GpuPreprocessKernels::camera_frame_to_model_tensor(&gpu_rgb_u8, &mut gpu_tensor, config)
            .unwrap();
    }
}
