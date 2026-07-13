use crate::{
    error::MediaConnectorError,
    video_sampling::{VideoFrameMetadata, VideoFrameSampler, VideoFrameSamplingTarget},
};

/// Qwen2/2.5-VL FPS sampler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen2VlFrameSampler {
    pub temporal_patch_size: usize,
}

impl VideoFrameSampler for Qwen2VlFrameSampler {
    fn sample_frame_indices(
        &self,
        source: VideoFrameMetadata,
        target: VideoFrameSamplingTarget,
    ) -> Result<Vec<usize>, MediaConnectorError> {
        let temporal_patch_size = self.temporal_patch_size;
        if temporal_patch_size == 0 {
            return Err(MediaConnectorError::VideoDecode(
                "Qwen2-VL temporal_patch_size must be greater than 0".to_string(),
            ));
        }
        let total = source.total_frames;
        let aligned_max = target.max_frames.min(total) / temporal_patch_size * temporal_patch_size;
        if aligned_max == 0 {
            return Err(MediaConnectorError::VideoDecode(
                "Qwen2-VL frame limit is smaller than temporal_patch_size".to_string(),
            ));
        }
        let count = (total as f64 / source.original_fps * target.fps)
            .max(target.min_frames as f64)
            .min(aligned_max as f64)
            .min(total as f64) as usize;
        let count = count / temporal_patch_size * temporal_patch_size;
        if count == 0 {
            return Err(MediaConnectorError::VideoDecode(
                "Qwen2-VL sampling produced zero frames".to_string(),
            ));
        }

        // transformers uses torch.arange with float32 index math.
        let step = total as f32 / count as f32;
        let mut indices = Vec::with_capacity(count + 1);
        for idx in 0..=count {
            let value = idx as f32 * step;
            if value >= total as f32 {
                break;
            }
            indices.push((value as usize).min(total - 1));
        }
        Ok(indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video_sampling::sample_video_frame_indices;

    #[test]
    fn aligns_to_temporal_patches() {
        let indices = sample_video_frame_indices(
            VideoFrameMetadata {
                total_frames: 100,
                original_fps: 25.0,
            },
            VideoFrameSamplingTarget {
                min_frames: 4,
                max_frames: 768,
                fps: 2.0,
            },
            &Qwen2VlFrameSampler {
                temporal_patch_size: 2,
            },
        )
        .unwrap();
        assert_eq!(indices, vec![0, 12, 25, 37, 50, 62, 75, 87]);
    }
}
