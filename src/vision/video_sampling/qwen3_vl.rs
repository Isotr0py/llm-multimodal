use crate::{
    error::MediaConnectorError,
    video_sampling::{
        UniformFpsFrameSampler, VideoFrameMetadata, VideoFrameSampler, VideoFrameSamplingTarget,
    },
};

/// Qwen3-VL uses the reusable uniform FPS sampling algorithm.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Qwen3VlFrameSampler;

impl VideoFrameSampler for Qwen3VlFrameSampler {
    fn sample_frame_indices(
        &self,
        source: VideoFrameMetadata,
        target: VideoFrameSamplingTarget,
    ) -> Result<Vec<usize>, MediaConnectorError> {
        UniformFpsFrameSampler.sample_frame_indices(source, target)
    }
}
