use std::{collections::HashMap, sync::Arc};

use tokio::task::JoinHandle;

use super::{
    error::{MultiModalError, MultiModalResult},
    media::{ImageFetchConfig, MediaConnector, MediaSource, VideoFetchConfig},
    registry::{ModelMetadata, ModelRegistry, RegistryResult},
    types::{
        ImageDetail, MediaContentPart, Modality, MultiModalData, MultiModalUUIDs, TrackedMedia,
    },
};

type PendingTask = JoinHandle<MultiModalResult<TrackedMedia>>;

#[derive(Debug)]
pub struct TrackerOutput {
    pub data: MultiModalData,
    pub uuids: MultiModalUUIDs,
}

pub struct AsyncMultiModalTracker {
    media_connector: Arc<MediaConnector>,
    video_fetch_config: VideoFetchConfig,
    pending: HashMap<Modality, Vec<PendingTask>>,
    uuids: MultiModalUUIDs,
}

impl AsyncMultiModalTracker {
    pub fn new(media_connector: Arc<MediaConnector>) -> Self {
        Self {
            media_connector,
            video_fetch_config: VideoFetchConfig::default(),
            pending: HashMap::new(),
            uuids: HashMap::new(),
        }
    }

    /// Create a tracker with video sampling selected from model metadata.
    pub fn new_for_model(
        media_connector: Arc<MediaConnector>,
        registry: &ModelRegistry,
        metadata: &ModelMetadata<'_>,
    ) -> RegistryResult<Self> {
        let video_fetch_config = registry.video_fetch_config(metadata)?;
        Ok(Self::new(media_connector).with_video_fetch_config(video_fetch_config))
    }

    /// Configure model-specific video frame sampling for subsequently queued
    /// video parts.
    pub fn with_video_fetch_config(mut self, config: VideoFetchConfig) -> Self {
        self.video_fetch_config = config;
        self
    }

    pub fn push_part(&mut self, part: MediaContentPart) -> MultiModalResult<()> {
        match part {
            MediaContentPart::Text { .. } => {}
            MediaContentPart::ImageUrl { url, detail, uuid } => {
                let source = match url::Url::parse(&url) {
                    Ok(parsed) if parsed.scheme() == "data" => MediaSource::DataUrl(url),
                    _ => MediaSource::Url(url),
                };
                self.enqueue_image(source, detail.unwrap_or_default(), uuid);
            }
            MediaContentPart::ImageData {
                data,
                mime_type: _,
                uuid,
                detail,
            } => {
                self.enqueue_image(
                    MediaSource::InlineBytes(data),
                    detail.unwrap_or_default(),
                    uuid,
                );
            }
            MediaContentPart::ImageEmbeds { .. } => {
                return Err(MultiModalError::UnsupportedContent("image_embeds"));
            }
            MediaContentPart::VideoUrl { url, uuid } => {
                let source = match url::Url::parse(&url) {
                    Ok(parsed) if parsed.scheme() == "data" => MediaSource::DataUrl(url),
                    _ => MediaSource::Url(url),
                };
                self.enqueue_video(source, uuid);
            }
            MediaContentPart::VideoData {
                data,
                mime_type: _,
                uuid,
            } => {
                self.enqueue_video(MediaSource::InlineBytes(data), uuid);
            }
        }
        Ok(())
    }

    pub async fn finalize(mut self) -> MultiModalResult<TrackerOutput> {
        let mut data = MultiModalData::new();
        for (modality, tasks) in self.pending.drain() {
            let mut items = Vec::with_capacity(tasks.len());
            for task in tasks {
                let media = task.await??;
                items.push(media);
            }
            data.insert(modality, items);
        }

        Ok(TrackerOutput {
            data,
            uuids: self.uuids,
        })
    }

    fn enqueue_image(&mut self, source: MediaSource, detail: ImageDetail, uuid: Option<String>) {
        let modality = Modality::Image;
        self.uuids.entry(modality).or_default().push(uuid);

        let connector = Arc::clone(&self.media_connector);
        let handle = tokio::spawn(async move {
            let frame = connector
                .fetch_image(source, ImageFetchConfig { detail })
                .await?;
            Ok(TrackedMedia::Image(frame))
        });

        self.pending.entry(modality).or_default().push(handle);
    }

    fn enqueue_video(&mut self, source: MediaSource, uuid: Option<String>) {
        let modality = Modality::Video;
        self.uuids.entry(modality).or_default().push(uuid);

        let connector = Arc::clone(&self.media_connector);
        let config = self.video_fetch_config.clone();
        let handle = tokio::spawn(async move {
            let clip = connector.fetch_video(source, config).await?;
            Ok(TrackedMedia::Video(clip))
        });

        self.pending.entry(modality).or_default().push(handle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        media::{sample_video_frame_indices, MediaConnectorConfig},
        registry::Tokenizer,
        video_sampling::VideoFrameMetadata,
    };
    use serde_json::json;

    struct EmptyTokenizer;

    impl Tokenizer for EmptyTokenizer {
        fn token_to_id(&self, _token: &str) -> Option<u32> {
            None
        }

        fn id_to_token(&self, _id: u32) -> Option<String> {
            None
        }

        fn encode_text(&self, _text: &str) -> Option<Vec<u32>> {
            None
        }
    }

    #[test]
    fn new_for_model_injects_registry_video_sampler() {
        let connector = Arc::new(
            MediaConnector::new(reqwest::Client::new(), MediaConnectorConfig::default()).unwrap(),
        );
        let tokenizer = EmptyTokenizer;
        let config = json!({
            "model_type": "qwen2_vl",
            "vision_config": {"temporal_patch_size": 4}
        });
        let metadata = ModelMetadata {
            model_id: "Qwen2-VL-7B",
            tokenizer: &tokenizer,
            config: &config,
        };
        let tracker =
            AsyncMultiModalTracker::new_for_model(connector, &ModelRegistry::new(), &metadata)
                .unwrap();

        let indices = sample_video_frame_indices(
            VideoFrameMetadata {
                total_frames: 75,
                original_fps: 25.0,
            },
            &tracker.video_fetch_config,
        )
        .unwrap();
        assert_eq!(indices, vec![0, 18, 37, 56]);
    }
}
