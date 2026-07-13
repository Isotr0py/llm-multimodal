//! Golden tests covering encoded video fetch, model sampling, and preprocessing.

#![expect(clippy::expect_used, reason = "golden test helpers")]
#![expect(clippy::print_stdout, reason = "golden test diagnostics")]
#![expect(clippy::print_stderr, reason = "optional decoder diagnostics")]

use std::{fs::File, path::Path, process::Command, sync::Arc, time::Duration};

use llm_multimodal::{
    vision::{
        ModelSpecificValue, PreProcessorConfig, Qwen2VLProcessor, Qwen3VLProcessor,
        VisionPreProcessor,
    },
    AsyncMultiModalTracker, MediaConnector, MediaConnectorConfig, MediaContentPart, Modality,
    ModelMetadata, ModelRegistry, Tokenizer, TrackedMedia,
};

const FIXTURE_DIR: &str = "tests/fixtures/video_golden";
const PIXEL_TOLERANCE: f32 = 0.1;

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

struct GoldenVideo {
    pixels: Vec<f32>,
    pixel_shape: Vec<usize>,
    grid_thw: Vec<i64>,
    frame_indices: Vec<i64>,
}

fn read_npz_array<T: npyz::Deserialize>(
    archive: &mut npyz::npz::NpzArchive<File>,
    name: &str,
) -> (Vec<T>, Vec<usize>) {
    let reader = archive
        .by_name(name)
        .expect("read golden npz")
        .unwrap_or_else(|| panic!("missing {name} in golden npz"));
    let shape = reader.shape().iter().map(|&dim| dim as usize).collect();
    let data = reader.into_vec().expect("read golden array");
    (data, shape)
}

fn load_golden(path: &Path) -> GoldenVideo {
    let file = File::open(path).expect("open video golden");
    let mut archive = npyz::npz::NpzArchive::new(file).expect("parse video golden npz");
    let (pixels, pixel_shape) = read_npz_array(&mut archive, "pixel_values");
    let (grid_thw, _) = read_npz_array(&mut archive, "video_grid_thw");
    let (frame_indices, _) = read_npz_array(&mut archive, "frame_indices");
    GoldenVideo {
        pixels,
        pixel_shape,
        grid_thw,
        frame_indices,
    }
}

fn command_available(command: &str) -> bool {
    Command::new(command)
        .arg("-version")
        .output()
        .is_ok_and(|output| output.status.success())
}

fn can_decode_video() -> bool {
    cfg!(feature = "opencv-video") || (command_available("ffmpeg") && command_available("ffprobe"))
}

async fn run_video_golden(model: &str) {
    let fixture_dir = Path::new(FIXTURE_DIR);
    let video_path = fixture_dir.join("qwen_sampling.mp4");
    let golden_path = fixture_dir.join(format!("golden_{model}.npz"));
    let config_path = fixture_dir.join(format!("preprocessor_config_{model}.json"));
    assert!(
        video_path.exists() && golden_path.exists() && config_path.exists(),
        "video golden fixture missing; run: python scripts/generate_video_golden.py"
    );
    if !can_decode_video() {
        eprintln!("video golden skipped: OpenCV feature or ffmpeg+ffprobe is required");
        return;
    }

    let connector = Arc::new(
        MediaConnector::new(
            reqwest::Client::builder()
                .no_proxy()
                .build()
                .expect("build HTTP client"),
            MediaConnectorConfig {
                allowed_domains: None,
                allowed_local_media_path: Some(
                    std::fs::canonicalize(fixture_dir).expect("canonical fixture directory"),
                ),
                fetch_timeout: Duration::from_secs(10),
            },
        )
        .expect("create media connector"),
    );
    let tokenizer = EmptyTokenizer;
    let raw_model_config = serde_json::json!({
        "model_type": model,
        "vision_config": {"temporal_patch_size": 2}
    });
    let metadata = ModelMetadata {
        model_id: model,
        tokenizer: &tokenizer,
        config: &raw_model_config,
    };
    let mut tracker =
        AsyncMultiModalTracker::new_for_model(connector, &ModelRegistry::new(), &metadata)
            .expect("select model video sampler");
    tracker
        .push_part(MediaContentPart::VideoData {
            data: std::fs::read(video_path).expect("read encoded video fixture"),
            mime_type: Some("video/mp4".to_string()),
            uuid: Some(format!("{model}-golden")),
        })
        .expect("queue video fixture");
    let tracked = tracker.finalize().await.expect("fetch and decode video");
    let video = match &tracked.data[&Modality::Video][0] {
        TrackedMedia::Video(video) => video,
        other => panic!("expected tracked video, got {other:?}"),
    };

    let config_json = std::fs::read_to_string(config_path).expect("read processor config");
    let config = PreProcessorConfig::from_json(&config_json).expect("parse processor config");
    let processor: Box<dyn VisionPreProcessor> = match model {
        "qwen2_vl" => Box::new(Qwen2VLProcessor::from_preprocessor_config(&config)),
        "qwen3_vl" => Box::new(Qwen3VLProcessor::from_preprocessor_config(&config)),
        _ => panic!("unsupported video golden model: {model}"),
    };
    let processed = if let Some(rgb_video) = video.rgb_video() {
        let refs = rgb_video.frame_refs().expect("borrow decoded RGB frames");
        processor
            .preprocess_video_rgb(&refs, &config)
            .expect("preprocess RGB video")
    } else {
        processor
            .preprocess_video(video.frames(), &config)
            .expect("preprocess video frames")
    };

    let golden = load_golden(&golden_path);
    let decoded_frames = video.frames().len()
        + video
            .rgb_video()
            .map_or(0, |rgb_video| rgb_video.frames.len());
    assert_eq!(
        decoded_frames,
        golden.frame_indices.len(),
        "{model}: sampled frame count differs from Transformers indices {:?}",
        golden.frame_indices
    );
    assert_eq!(
        processed.encoder_input.shape(),
        golden.pixel_shape,
        "{model}: pixel_values shape mismatch"
    );
    let rust_grid = match processed.model_specific.get("video_grid_thw") {
        Some(ModelSpecificValue::IntTensor { data, shape }) => {
            assert_eq!(shape, &[1, 3]);
            data
        }
        other => panic!("{model}: missing video_grid_thw, got {other:?}"),
    };
    assert_eq!(rust_grid, &golden.grid_thw, "{model}: video grid mismatch");

    let max_diff = processed
        .encoder_input
        .iter()
        .zip(&golden.pixels)
        .map(|(&rust, &expected)| (rust - expected).abs())
        .fold(0.0_f32, f32::max);
    println!(
        "{model}: frames={}, shape={:?}, grid={:?}, max difference={max_diff:.6}",
        golden.frame_indices.len(),
        golden.pixel_shape,
        golden.grid_thw
    );
    assert!(
        max_diff < PIXEL_TOLERANCE,
        "{model}: max pixel difference {max_diff} exceeds {PIXEL_TOLERANCE}"
    );
}

#[tokio::test]
async fn qwen2_vl_complete_video_pipeline_matches_transformers() {
    run_video_golden("qwen2_vl").await;
}

#[tokio::test]
async fn qwen3_vl_complete_video_pipeline_matches_transformers() {
    run_video_golden("qwen3_vl").await;
}
