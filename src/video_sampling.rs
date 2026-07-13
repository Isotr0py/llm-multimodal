use crate::error::MediaConnectorError;

/// Source metadata needed by frame samplers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VideoFrameMetadata {
    pub total_frames: usize,
    pub original_fps: f64,
}

/// Model-requested frame sampling limits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VideoFrameSamplingTarget {
    pub min_frames: usize,
    pub max_frames: usize,
    pub fps: f64,
}

/// Extensible source-frame sampling interface.
pub trait VideoFrameSampler: std::fmt::Debug + Send + Sync {
    /// Return non-decreasing source-frame indices. Adjacent duplicates are
    /// allowed when a model requires frame padding.
    fn sample_frame_indices(
        &self,
        source: VideoFrameMetadata,
        target: VideoFrameSamplingTarget,
    ) -> Result<Vec<usize>, MediaConnectorError>;
}

/// Reusable uniform FPS sampler used by Qwen3-VL and compatible models.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UniformFpsFrameSampler;

impl VideoFrameSampler for UniformFpsFrameSampler {
    fn sample_frame_indices(
        &self,
        source: VideoFrameMetadata,
        target: VideoFrameSamplingTarget,
    ) -> Result<Vec<usize>, MediaConnectorError> {
        let count = ((source.total_frames as f64 / source.original_fps * target.fps) as usize)
            .clamp(target.min_frames, target.max_frames)
            .min(source.total_frames);
        Ok(uniform_indices_round(source.total_frames, count.max(1)))
    }
}

/// Run a sampler and validate its decoder-facing index contract.
pub fn sample_video_frame_indices(
    source: VideoFrameMetadata,
    target: VideoFrameSamplingTarget,
    sampler: &dyn VideoFrameSampler,
) -> Result<Vec<usize>, MediaConnectorError> {
    if source.total_frames == 0 {
        return Err(sampling_error("video contains no frames"));
    }
    require_positive_fps(source.original_fps, "video")?;
    require_positive_fps(target.fps, "target")?;
    if target.min_frames == 0 || target.max_frames == 0 {
        return Err(sampling_error(
            "video sampling frame limits must be greater than 0",
        ));
    }
    if target.min_frames > target.max_frames {
        return Err(sampling_error(
            "min_frames must be less than or equal to max_frames",
        ));
    }

    let indices = sampler.sample_frame_indices(source, target)?;
    if indices.is_empty() {
        return Err(sampling_error("video sampling produced no frames"));
    }
    if let Some(index) = indices.iter().find(|index| **index >= source.total_frames) {
        return Err(sampling_error(format!(
            "video sampler returned out-of-range frame index {index} for {} frames",
            source.total_frames
        )));
    }
    if indices.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(sampling_error(
            "video sampler must return non-decreasing frame indices",
        ));
    }
    Ok(indices)
}

fn sampling_error(message: impl Into<String>) -> MediaConnectorError {
    MediaConnectorError::VideoDecode(message.into())
}

fn require_positive_fps(fps: f64, name: &str) -> Result<(), MediaConnectorError> {
    if fps.is_finite() && fps > 0.0 {
        Ok(())
    } else {
        Err(sampling_error(format!(
            "{name} sampling needs a known positive fps"
        )))
    }
}

fn linspace_values(start: f64, end: f64, count: usize) -> Vec<f64> {
    match count {
        0 => Vec::new(),
        1 => vec![start],
        _ => (0..count)
            .map(|idx| start + (end - start) * idx as f64 / (count - 1) as f64)
            .collect(),
    }
}

fn uniform_indices_round(total: usize, count: usize) -> Vec<usize> {
    linspace_values(0.0, (total - 1) as f64, count)
        .into_iter()
        .map(|value| value.round_ties_even().max(0.0) as usize)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct FirstLastFrameSampler;

    impl VideoFrameSampler for FirstLastFrameSampler {
        fn sample_frame_indices(
            &self,
            source: VideoFrameMetadata,
            _target: VideoFrameSamplingTarget,
        ) -> Result<Vec<usize>, MediaConnectorError> {
            Ok(vec![0, source.total_frames - 1])
        }
    }

    #[test]
    fn custom_sampler_plugs_in_without_core_changes() {
        let indices = sample_video_frame_indices(
            VideoFrameMetadata {
                total_frames: 10,
                original_fps: 25.0,
            },
            VideoFrameSamplingTarget {
                min_frames: 1,
                max_frames: 10,
                fps: 2.0,
            },
            &FirstLastFrameSampler,
        )
        .unwrap();
        assert_eq!(indices, vec![0, 9]);
    }

    #[test]
    fn uniform_fps_matches_qwen3_rounding() {
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
            &UniformFpsFrameSampler,
        )
        .unwrap();
        assert_eq!(indices, vec![0, 14, 28, 42, 57, 71, 85, 99]);
    }
}
