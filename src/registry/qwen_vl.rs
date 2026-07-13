use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    media::VideoFetchConfig,
    registry::{ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult},
    types::{FieldLayout, Modality, PromptReplacement, TokenId},
    vision::{processor::PreprocessedEncoderInputs, video_sampling::Qwen2VlFrameSampler},
};

pub(super) struct QwenVLVisionSpec;

impl QwenVLVisionSpec {
    fn pad_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["image_token_id"])
            .map(|v| v as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "image_token_id".to_string(),
            })
    }
}

impl ModelProcessorSpec for QwenVLVisionSpec {
    fn name(&self) -> &'static str {
        "qwen_vl"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let id = metadata.model_id.to_ascii_lowercase();
        id.contains("qwen") && id.contains("vl")
            || metadata
                .config_model_type()
                .is_some_and(|mt| mt == "qwen2_vl")
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok("<|image_pad|>".to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        // Must match pad_token_id (vision_token_id) — this is the repeated token
        // in the expanded sequence. image_token_id is a distinct token in Qwen2-VL.
        Self::pad_token_id(metadata)
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, 10)]))
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn video_fetch_config(
        &self,
        metadata: &ModelMetadata,
    ) -> RegistryResult<Option<VideoFetchConfig>> {
        let temporal_patch_size = metadata
            .config_u32(&["vision_config", "temporal_patch_size"])
            .unwrap_or(2) as usize;
        Ok(Some(VideoFetchConfig::default().with_frame_sampler(
            Qwen2VlFrameSampler {
                temporal_patch_size,
            },
        )))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let pad_token_id = Self::pad_token_id(metadata)?;
        let placeholder_token = self.placeholder_token(metadata)?;
        // The chat template already wraps each image with <|vision_start|> ... <|vision_end|>,
        // so we only expand the single <image> placeholder to N pad tokens.
        Ok(preprocessed
            .feature_token_counts
            .iter()
            .map(|&num_tokens| {
                let tokens = vec![pad_token_id; num_tokens];
                PromptReplacement::sequence(Modality::Image, &placeholder_token, tokens)
            })
            .collect())
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        // encoder_input is patchified: [total_patches, patch_features].
        // patches_per_image tells how many patches belong to each image.
        // image_grid_thw is [num_images, 3].
        HashMap::from([
            (
                "pixel_values".to_string(),
                FieldLayout::flat("patches_per_image"),
            ),
            ("image_grid_thw".to_string(), FieldLayout::Batched),
            ("patches_per_image".to_string(), FieldLayout::Batched),
        ])
    }

    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec!["image_grid_thw".to_string()]
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::{
        media::sample_video_frame_indices,
        registry::{test_helpers::*, ModelMetadata, ModelRegistry},
        types::ImageSize,
        video_sampling::VideoFrameMetadata,
    };

    #[test]
    fn qwen_vision_uses_config_token_ids() {
        let tokenizer = TestTokenizer::new(&[("<|image_pad|>", 151655)]);
        let config = json!({
            "model_type": "qwen2_vl",
            "vision_start_token_id": 151652,
            "vision_token_id": 151654,
            "image_token_id": 151655,
            "vision_config": {"patch_size": 14}
        });
        let metadata = ModelMetadata {
            model_id: "Qwen2-VL-7B",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("qwen spec");
        // 448/14 = 32 grid, merge_size=2 => (32*32)/4 = 256 tokens
        let replacements = spec
            .prompt_replacements(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(448, 448)], &[256]),
            )
            .unwrap();
        // Only pad tokens — vision_start/vision_end are already in the chat template
        assert_eq!(replacements[0].tokens.len(), 256);
        assert_eq!(replacements[0].tokens[0], 151655); // pad (image_token_id)
        assert_eq!(*replacements[0].tokens.last().unwrap(), 151655);
    }

    #[test]
    fn qwen_vl_matches_alias_via_model_type() {
        let tokenizer = TestTokenizer::new(&[("<|image_pad|>", 151655)]);
        let config = json!({
            "model_type": "qwen2_vl",
            "vision_start_token_id": 151652,
            "vision_token_id": 151654,
            "image_token_id": 151655
        });
        let metadata = ModelMetadata {
            model_id: "custom-model",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("should match qwen alias");
        assert_eq!(spec.name(), "qwen_vl");
    }

    #[test]
    fn qwen2_registry_injects_temporal_patch_sampler() {
        let tokenizer = TestTokenizer::new(&[]);
        let config = json!({
            "model_type": "qwen2_vl",
            "vision_config": {"temporal_patch_size": 4}
        });
        let metadata = ModelMetadata {
            model_id: "Qwen2-VL-7B",
            tokenizer: &tokenizer,
            config: &config,
        };
        let video_config = ModelRegistry::new().video_fetch_config(&metadata).unwrap();
        let indices = sample_video_frame_indices(
            VideoFrameMetadata {
                total_frames: 75,
                original_fps: 25.0,
            },
            &video_config,
        )
        .unwrap();
        assert_eq!(indices, vec![0, 18, 37, 56]);
    }
}
