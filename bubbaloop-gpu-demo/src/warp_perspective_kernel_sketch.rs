//! GPU warp_perspective dispatch sketch for proposal architecture.
//! This file is standalone and not wired into main.rs.

pub struct CpuAllocator;
pub struct GpuAllocator {
    pub device_id: u32,
}

pub struct Image<T, const C: usize, A> {
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

pub trait WarpPerspectiveKernel<const C: usize, A: ImageAllocator> {
    fn warp_perspective_f32(
        src: &Image<f32, C, A>,
        dst: &mut Image<f32, C, A>,
        m: &[f32; 9],
    ) -> Result<(), ImageError>
    where
        Self: Sized;
}

pub struct CpuWarpPerspective;

impl<const C: usize> WarpPerspectiveKernel<C, CpuAllocator> for CpuWarpPerspective {
    fn warp_perspective_f32(
        _src: &Image<f32, C, CpuAllocator>,
        _dst: &mut Image<f32, C, CpuAllocator>,
        _m: &[f32; 9],
    ) -> Result<(), ImageError> {
        Ok(())
    }
}

pub struct GpuWarpPerspective;

impl<const C: usize> WarpPerspectiveKernel<C, GpuAllocator> for GpuWarpPerspective {
    fn warp_perspective_f32(
        _src: &Image<f32, C, GpuAllocator>,
        _dst: &mut Image<f32, C, GpuAllocator>,
        _m: &[f32; 9],
    ) -> Result<(), ImageError> {
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_warp_compiles_and_runs() {
        let src: Image<f32, 3, CpuAllocator> = Image {
            _marker: std::marker::PhantomData,
        };
        let mut dst: Image<f32, 3, CpuAllocator> = Image {
            _marker: std::marker::PhantomData,
        };
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0f32];
        CpuWarpPerspective::warp_perspective_f32(&src, &mut dst, &identity).unwrap();
    }

    #[test]
    fn gpu_warp_compiles_and_runs() {
        let src: Image<f32, 3, GpuAllocator> = Image {
            _marker: std::marker::PhantomData,
        };
        let mut dst: Image<f32, 3, GpuAllocator> = Image {
            _marker: std::marker::PhantomData,
        };
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0f32];
        GpuWarpPerspective::warp_perspective_f32(&src, &mut dst, &identity).unwrap();
    }

    #[test]
    fn explicit_transfer_compiles() {
        let cpu_img: Image<f32, 3, CpuAllocator> = Image {
            _marker: std::marker::PhantomData,
        };
        let gpu_img = cpu_img.to_device(GpuAllocator { device_id: 0 }).unwrap();
        let _back = gpu_img.to_host(CpuAllocator).unwrap();
    }
}
