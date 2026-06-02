use crate::{Result, Tensor};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
struct ArgSort {
    asc: bool,
    last_dim: usize,
}

impl ArgSort {
    fn asort<T: crate::WithDType>(&self, vs: &[T], layout: &crate::Layout) -> Vec<u32> {
        #[allow(clippy::uninit_vec)]
        // Safety: indexes are set later in the parallelized section.
        let mut sort_indexes = unsafe {
            let el_count = layout.shape().elem_count();
            let mut v = Vec::with_capacity(el_count);
            v.set_len(el_count);
            v
        };
        if self.asc {
            sort_indexes
                .par_chunks_exact_mut(self.last_dim)
                .zip(vs.par_chunks_exact(self.last_dim))
                .for_each(|(indexes, vs)| {
                    indexes
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, v)| *v = i as u32);
                    indexes.sort_by(|&i, &j| {
                        vs[i as usize]
                            .partial_cmp(&vs[j as usize])
                            .unwrap_or(std::cmp::Ordering::Greater)
                    })
                });
        } else {
            sort_indexes
                .par_chunks_exact_mut(self.last_dim)
                .zip(vs.par_chunks_exact(self.last_dim))
                .for_each(|(indexes, vs)| {
                    indexes
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, v)| *v = i as u32);
                    indexes.sort_by(|&j, &i| {
                        vs[i as usize]
                            .partial_cmp(&vs[j as usize])
                            .unwrap_or(std::cmp::Ordering::Greater)
                    })
                });
        }
        sort_indexes
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use crate::cuda_backend::cudarc::driver::{
        CudaSlice, DeviceRepr, LaunchConfig, ValidAsZeroBits,
    };
    use crate::cuda_backend::{kernel_name, kernels, CudaStorageSlice as S, WrapErr};
    use crate::{CudaDevice, WithDType};

    impl crate::cuda_backend::Map1Any for ArgSort {
        fn f<T: DeviceRepr + WithDType + ValidAsZeroBits, W: Fn(CudaSlice<T>) -> S>(
            &self,
            src: &CudaSlice<T>,
            dev: &CudaDevice,
            layout: &crate::Layout,
            _wrap: W,
        ) -> Result<S> {
            use cudarc::driver::PushKernelArg;

            let slice = match layout.contiguous_offsets() {
                None => crate::bail!("input has to be contiguous"),
                Some((o1, o2)) => src.slice(o1..o2),
            };
            let elem_count = layout.shape().elem_count();
            let dst = unsafe { dev.alloc::<u32>(elem_count)? };
            let func = if self.asc {
                dev.get_or_load_func(&kernel_name::<T>("asort_asc"), &kernels::SORT)?
            } else {
                dev.get_or_load_func(&kernel_name::<T>("asort_desc"), &kernels::SORT)?
            };
            let ncols = self.last_dim;
            let nrows = elem_count / ncols;
            let ncols_pad = next_power_of_2(ncols);
            // Limit block dim to 1024 threads, which is the maximum on modern CUDA gpus.
            let block_dim = ncols_pad.min(1024);
            let cfg = LaunchConfig {
                grid_dim: (nrows as u32, 1, 1),
                block_dim: (block_dim as u32, 1, 1),
                shared_mem_bytes: (ncols_pad * std::mem::size_of::<u32>()) as u32,
            };
            let stream = dev.cuda_stream();
            let mut builder = stream.launch_builder(&func);
            let ncols = ncols as i32;
            let ncols_pad = ncols_pad as i32;
            builder.arg(&slice).arg(&dst).arg(&ncols).arg(&ncols_pad);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(S::U32(dst))
        }
    }
}

impl crate::CustomOp1 for ArgSort {
    fn name(&self) -> &'static str {
        "argsort"
    }

    // GPU-native bitonic argsort over the last dim (the MoE routing hot path, cols == num_experts).
    // Only f32 rows within the kernel's shared-scratch bound run on the GPU; u32 inputs (e.g. sorting
    // expert ids) and over-wide rows take the CPU roundtrip, which is correct and off the hot path.
    #[cfg(feature = "vulkan")]
    fn vulkan_fwd(
        &self,
        storage: &crate::VulkanStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::VulkanStorage, crate::Shape)> {
        use crate::backend::{BackendDevice, BackendStorage};
        if storage.dtype() == crate::DType::F32 {
            if let Some(out) = storage.arg_sort_last_dim(layout, self.asc, self.last_dim)? {
                return Ok((out, layout.shape().into()));
            }
        }
        let cpu = storage.to_cpu_storage()?;
        let (sorted, shape) = self.cpu_fwd(&cpu, layout)?;
        let out = storage.device().storage_from_cpu_storage(&sorted)?;
        Ok((out, shape))
    }

    fn cpu_fwd(
        &self,
        storage: &crate::CpuStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CpuStorage, crate::Shape)> {
        let sort_indexes = match storage {
            crate::CpuStorage::U8(vs) => self.asort(vs, layout),
            crate::CpuStorage::U32(vs) => self.asort(vs, layout),
            crate::CpuStorage::I16(vs) => self.asort(vs, layout),
            crate::CpuStorage::I32(vs) => self.asort(vs, layout),
            crate::CpuStorage::I64(vs) => self.asort(vs, layout),
            crate::CpuStorage::BF16(vs) => self.asort(vs, layout),
            crate::CpuStorage::F16(vs) => self.asort(vs, layout),
            crate::CpuStorage::F32(vs) => self.asort(vs, layout),
            crate::CpuStorage::F64(vs) => self.asort(vs, layout),
            crate::CpuStorage::F8E4M3(vs) => self.asort(vs, layout),
            // Dummy types don't support sorting
            crate::CpuStorage::F6E2M3(_) => {
                return Err(
                    crate::Error::UnsupportedDTypeForOp(crate::DType::F6E2M3, "argsort").bt(),
                )
            }
            crate::CpuStorage::F6E3M2(_) => {
                return Err(
                    crate::Error::UnsupportedDTypeForOp(crate::DType::F6E3M2, "argsort").bt(),
                )
            }
            crate::CpuStorage::F4(_) => {
                return Err(crate::Error::UnsupportedDTypeForOp(crate::DType::F4, "argsort").bt())
            }
            crate::CpuStorage::F8E8M0(_) => {
                return Err(
                    crate::Error::UnsupportedDTypeForOp(crate::DType::F8E8M0, "argsort").bt(),
                )
            }
        };
        let sort_indexes = crate::CpuStorage::U32(sort_indexes);
        Ok((sort_indexes, layout.shape().into()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &crate::CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CudaStorage, crate::Shape)> {
        use crate::backend::BackendStorage;
        use crate::cuda_backend::Map1Any;
        let dev = storage.device();
        let slice = self.map(&storage.slice, dev, layout)?;
        let dst = crate::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, layout.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &crate::MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::MetalStorage, crate::Shape)> {
        use crate::backend::BackendStorage;
        use crate::DType;

        let name = {
            if self.asc {
                match storage.dtype() {
                    DType::BF16 => "asort_asc_bf16",
                    DType::F16 => "asort_asc_f16",
                    DType::F32 => "asort_asc_f32",
                    DType::F64 => "asort_asc_f64",
                    DType::U8 => "asort_asc_u8",
                    DType::U32 => "asort_asc_u32",
                    DType::I16 => "asort_asc_i16",
                    DType::I32 => "asort_asc_i32",
                    DType::I64 => "asort_asc_i64",
                    DType::F8E4M3 => crate::bail!("Metal device does not yet support F8E4M3."),
                    DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                        return Err(
                            crate::Error::UnsupportedDTypeForOp(storage.dtype(), "argsort").bt(),
                        )
                    }
                }
            } else {
                match storage.dtype() {
                    DType::BF16 => "asort_desc_bf16",
                    DType::F16 => "asort_desc_f16",
                    DType::F32 => "asort_desc_f32",
                    DType::F64 => "asort_desc_f64",
                    DType::U8 => "asort_desc_u8",
                    DType::U32 => "asort_desc_u32",
                    DType::I16 => "asort_desc_i16",
                    DType::I32 => "asort_desc_i32",
                    DType::I64 => "asort_desc_i64",
                    DType::F8E4M3 => crate::bail!("Metal device does not yet support F8E4M3."),
                    DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                        return Err(
                            crate::Error::UnsupportedDTypeForOp(storage.dtype(), "argsort").bt(),
                        )
                    }
                }
            }
        };
        let device = storage.device();
        let kernels = device.kernels();
        let command_encoder = device.command_encoder()?;
        let el = layout.shape().elem_count();
        let ncols = self.last_dim;
        let nrows = el / ncols;
        let src = crate::metal_backend::buffer_o(storage.buffer(), layout, storage.dtype());
        let dst = device.new_buffer(el, DType::U32, "asort")?;
        let mut ncols_pad = 1;
        while ncols_pad < ncols {
            ncols_pad *= 2;
        }
        hanzo_metal_kernels::call_arg_sort(
            device.metal_device(),
            &command_encoder,
            kernels,
            name,
            nrows,
            ncols,
            ncols_pad,
            src,
            &dst,
        )
        .map_err(crate::Error::wrap)?;
        let dst = crate::MetalStorage::new(dst, device.clone(), el, DType::U32);
        Ok((dst, layout.shape().clone()))
    }

    #[cfg(feature = "rocm")]
    fn rocm_fwd(
        &self,
        storage: &crate::RocmStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::RocmStorage, crate::Shape)> {
        let dst = storage.asort(layout, self.asc, self.last_dim)?;
        Ok((dst, layout.shape().clone()))
    }
}

#[allow(unused)]
fn next_power_of_2(x: usize) -> usize {
    let mut n = 1;
    while n < x {
        n *= 2
    }
    n
}

// CPU mirror of argsort.comp / the CUDA k_argsort bitonic network. Lets the unit test below check
// the exact comparator + padding semantics the GPU kernel implements, with no GPU. `asc` matches the
// shader's `asc == ASC` branch; out-of-range padded indices (>= cols) sort to the high end.
#[cfg(test)]
fn bitonic_argsort_ref(vals: &[f32], asc: bool) -> Vec<u32> {
    let cols = vals.len();
    let cols_pad = next_power_of_2(cols);
    let mut idx: Vec<usize> = (0..cols_pad).collect();
    // literal transcription of the shader's two comparator branches (a == dst_row[col],
    // b == dst_row[ixj]); `gt`/`lt` use the same strict tests so ties resolve identically.
    let gt = |a: usize, b: usize| if asc { vals[a] > vals[b] } else { vals[a] < vals[b] };
    let lt = |a: usize, b: usize| if asc { vals[a] < vals[b] } else { vals[a] > vals[b] };
    let mut k = 2;
    while k <= cols_pad {
        let mut j = k / 2;
        while j > 0 {
            for col in 0..cols_pad {
                let ixj = col ^ j;
                if ixj > col {
                    let (a, b) = (idx[col], idx[ixj]);
                    let swap = if col & k == 0 {
                        a >= cols || (b < cols && gt(a, b))
                    } else {
                        b >= cols || (a < cols && lt(a, b))
                    };
                    if swap {
                        idx.swap(col, ixj);
                    }
                }
            }
            j /= 2;
        }
        k *= 2;
    }
    idx[..cols].iter().map(|&i| i as u32).collect()
}

impl Tensor {
    /// Returns the indices that sort the tensor along the last dimension.
    ///
    /// If `asc` is `true`, sorting is in ascending order. Otherwise sorting is performed in
    /// descending order. The sort is unstable so there is no guarantees on the final order when it
    /// comes to ties.
    pub fn arg_sort_last_dim(&self, asc: bool) -> Result<Tensor> {
        if !self.is_contiguous() {
            return Err(crate::Error::RequiresContiguous {
                op: "arg_sort_last_dim",
            });
        }
        let last_dim = match self.dims().last() {
            None => crate::bail!("empty last-dim in arg-sort"),
            Some(last_dim) => *last_dim,
        };
        // No need for a backward pass for arg sort.
        self.apply_op1_no_bwd(&ArgSort { asc, last_dim })
    }

    /// Sorts the tensor along the last dimension, returns the sorted tensor together with the
    /// sorted indexes.
    ///
    /// If `asc` is `true`, sorting is in ascending order. Otherwise sorting is performed in
    /// descending order. The sort is unstable so there is no guarantees on the final order when it
    /// comes to ties.
    pub fn sort_last_dim(&self, asc: bool) -> Result<(Tensor, Tensor)> {
        if !self.is_contiguous() {
            return Err(crate::Error::RequiresContiguous {
                op: "sort_last_dim",
            });
        }
        let asort = self.arg_sort_last_dim(asc)?;
        let sorted = self.gather(&asort, crate::D::Minus1)?;
        Ok((sorted, asort))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The bitonic argsort the Vulkan kernel runs must match the CPU reference top-k: descending
    // values in order, and the first-k indices select the k largest. Ties may pick either index.
    #[test]
    fn bitonic_argsort_matches_reference_topk() {
        let rows: [Vec<f32>; 4] = [
            (0..128).map(|i| ((i * 37) % 101) as f32 * 0.5).collect(),
            vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0],
            vec![2.0, 2.0, 2.0, 1.0, 3.0, 3.0], // ties + non-power-of-2 width
            vec![-1.0, -5.0, 0.0, 10.0, 10.0],
        ];
        for vals in rows {
            let cols = vals.len();
            let order = bitonic_argsort_ref(&vals, false);
            assert_eq!(order.len(), cols);
            // every column index appears exactly once
            let mut seen = order.clone();
            seen.sort_unstable();
            assert_eq!(seen, (0..cols as u32).collect::<Vec<_>>());
            // gathered values are non-increasing (descending sort)
            let gathered: Vec<f32> = order.iter().map(|&i| vals[i as usize]).collect();
            for w in gathered.windows(2) {
                assert!(w[0] >= w[1], "not descending: {gathered:?}");
            }
            // top-k values equal the reference's k largest, for every k
            let mut ref_sorted = vals.clone();
            ref_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            for k in 1..=cols {
                assert_eq!(&gathered[..k], &ref_sorted[..k], "top-{k} value mismatch");
            }
        }
    }

    // Ascending direction sorts values non-decreasing (used by topk_unsorted's reorder pass).
    #[test]
    fn bitonic_argsort_ascending() {
        let vals = vec![5.0f32, 1.0, 4.0, 2.0, 3.0, 0.0, 6.0];
        let order = bitonic_argsort_ref(&vals, true);
        let gathered: Vec<f32> = order.iter().map(|&i| vals[i as usize]).collect();
        for w in gathered.windows(2) {
            assert!(w[0] <= w[1], "not ascending: {gathered:?}");
        }
    }
}
