# Copyright      2022  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from functools import lru_cache
from typing import Union

import torch
import torchaudio
from huggingface_hub import hf_hub_download

os.system(
    "cp -v /usr/local/lib/python3.8/site-packages/k2/lib/*.so //usr/local/lib/python3.8/site-packages/sherpa/lib/"
)

os.system(
    "cp -v /home/user/.local/lib/python3.8/site-packages/k2/lib/*.so /home/user/.local/lib/python3.8/site-packages/sherpa/lib/"
)

import k2  # noqa
import sherpa
import sherpa_onnx
import numpy as np
from typing import Tuple
import wave

sample_rate = 16000


def read_wave(wave_filename: str) -> Tuple[np.ndarray, int]:
    """
    Args:
      wave_filename:
        Path to a wave file. It should be single channel and each sample should
        be 16-bit. Its sample rate does not need to be 16kHz.
    Returns:
      Return a tuple containing:
       - A 1-D array of dtype np.float32 containing the samples, which are
       normalized to the range [-1, 1].
       - sample rate of the wave file
    """

    with wave.open(wave_filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)

        samples_float32 = samples_float32 / 32768
        return samples_float32, f.getframerate()


def decode_offline_recognizer(
    recognizer: sherpa.OfflineRecognizer,
    filename: str,
) -> str:
    s = recognizer.create_stream()

    s.accept_wave_file(filename)
    recognizer.decode_stream(s)

    text = s.result.text.strip()
    #  return text.lower()
    return text


def decode_online_recognizer(
    recognizer: sherpa.OnlineRecognizer,
    filename: str,
) -> str:
    samples, actual_sample_rate = torchaudio.load(filename)
    assert sample_rate == actual_sample_rate, (
        sample_rate,
        actual_sample_rate,
    )
    samples = samples[0].contiguous()

    s = recognizer.create_stream()

    tail_padding = torch.zeros(int(sample_rate * 0.3), dtype=torch.float32)
    s.accept_waveform(sample_rate, samples)
    s.accept_waveform(sample_rate, tail_padding)
    s.input_finished()

    while recognizer.is_ready(s):
        recognizer.decode_stream(s)

    text = recognizer.get_result(s).text
    #  return text.strip().lower()
    return text.strip()


def decode_offline_recognizer_sherpa_onnx(
    recognizer: sherpa_onnx.OfflineRecognizer,
    filename: str,
) -> str:
    s = recognizer.create_stream()
    samples, sample_rate = read_wave(filename)
    s.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(s)

    #  return s.result.text.lower()
    return s.result.text


def decode_online_recognizer_sherpa_onnx(
    recognizer: sherpa_onnx.OnlineRecognizer,
    filename: str,
) -> str:
    s = recognizer.create_stream()
    samples, sample_rate = read_wave(filename)
    s.accept_waveform(sample_rate, samples)

    tail_paddings = np.zeros(int(0.3 * sample_rate), dtype=np.float32)
    s.accept_waveform(sample_rate, tail_paddings)
    s.input_finished()

    while recognizer.is_ready(s):
        recognizer.decode_stream(s)

    #  return recognizer.get_result(s).lower()
    return recognizer.get_result(s)


def decode(
    recognizer: Union[
        sherpa.OfflineRecognizer,
        sherpa.OnlineRecognizer,
        sherpa_onnx.OfflineRecognizer,
        sherpa_onnx.OnlineRecognizer,
    ],
    filename: str,
) -> str:
    if isinstance(recognizer, sherpa.OfflineRecognizer):
        return decode_offline_recognizer(recognizer, filename)
    elif isinstance(recognizer, sherpa.OnlineRecognizer):
        return decode_online_recognizer(recognizer, filename)
    elif isinstance(recognizer, sherpa_onnx.OfflineRecognizer):
        return decode_offline_recognizer_sherpa_onnx(recognizer, filename)
    elif isinstance(recognizer, sherpa_onnx.OnlineRecognizer):
        return decode_online_recognizer_sherpa_onnx(recognizer, filename)
    else:
        raise ValueError(f"Unknown recognizer type {type(recognizer)}")


@lru_cache(maxsize=30)
def get_pretrained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
) -> Union[sherpa.OfflineRecognizer, sherpa.OnlineRecognizer]:
    if repo_id in chinese_models:
        return chinese_models[repo_id](
            repo_id, decoding_method=decoding_method, num_active_paths=num_active_paths
        )
    elif repo_id in chinese_dialect_models:
        return chinese_dialect_models[repo_id](
            repo_id, decoding_method=decoding_method, num_active_paths=num_active_paths
        )
    elif repo_id in english_models:
        return english_models[repo_id](
            repo_id, decoding_method=decoding_method, num_active_paths=num_active_paths
        )
    elif repo_id in chinese_english_mixed_models:
        return chinese_english_mixed_models[repo_id](
            repo_id, decoding_method=decoding_method, num_active_paths=num_active_paths
        )
    elif repo_id in chinese_cantonese_english_models:
        return chinese_cantonese_english_models[repo_id](
            repo_id, decoding_method=decoding_method, num_active_paths=num_active_paths
        )
    elif repo_id in chinese_cantonese_english_japanese_korean_models:
        return chinese_cantonese_english_japanese_korean_models[repo_id](
            repo_id, decoding_method=decoding_method, num_active_paths=num_active_paths
        )
    elif repo_id in cantonese_models:
        return cantonese_models[repo_id](
            repo_id, decoding_method=decoding_method, num_active_paths=num_active_paths
        )
    elif repo_id in tibetan_models:
        return tibetan_models[repo_id](
            repo_id, decoding_method=decoding_method, num_active_paths=num_active_paths
        )
    elif repo_id in arabic_models:
        return arabic_models[repo_id](
            repo_id, decoding_method=decoding_method, num_active_paths=num_active_paths
        )
    elif repo_id in german_models:
        return german_models[repo_id](
            repo_id, decoding_method=decoding_method, num_active_paths=num_active_paths
        )
    elif repo_id in french_models:
        return french_models[repo_id](
            repo_id, decoding_method=decoding_method, num_active_paths=num_active_paths
        )
    elif repo_id in japanese_models:
        return japanese_models[repo_id](
            repo_id, decoding_method=decoding_method, num_active_paths=num_active_paths
        )
    elif repo_id in russian_models:
        return russian_models[repo_id](
            repo_id, decoding_method=decoding_method, num_active_paths=num_active_paths
        )
    elif repo_id in korean_models:
        return korean_models[repo_id](
            repo_id, decoding_method=decoding_method, num_active_paths=num_active_paths
        )
    elif repo_id in thai_models:
        return thai_models[repo_id](
            repo_id, decoding_method=decoding_method, num_active_paths=num_active_paths
        )
    else:
        raise ValueError(f"Unsupported repo_id: {repo_id}")


def _get_nn_model_filename(
    repo_id: str,
    filename: str,
    subfolder: str = "exp",
) -> str:
    nn_model_filename = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
    )
    return nn_model_filename


def _get_bpe_model_filename(
    repo_id: str,
    filename: str = "bpe.model",
    subfolder: str = "data/lang_bpe_500",
) -> str:
    bpe_model_filename = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
    )
    return bpe_model_filename


def _get_token_filename(
    repo_id: str,
    filename: str = "tokens.txt",
    subfolder: str = "data/lang_char",
) -> str:
    token_filename = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
    )
    return token_filename


@lru_cache(maxsize=10)
def _get_aishell2_pretrained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
) -> sherpa.OfflineRecognizer:
    assert repo_id in [
        # context-size 1
        "yuekai/icefall-asr-aishell2-pruned-transducer-stateless5-A-2022-07-12",  # noqa
        # context-size 2
        "yuekai/icefall-asr-aishell2-pruned-transducer-stateless5-B-2022-07-12",  # noqa
    ], repo_id

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="cpu_jit.pt",
    )
    tokens = _get_token_filename(repo_id=repo_id)

    feat_config = sherpa.FeatureConfig()
    feat_config.fbank_opts.frame_opts.samp_freq = sample_rate
    feat_config.fbank_opts.mel_opts.num_bins = 80
    feat_config.fbank_opts.frame_opts.dither = 0

    config = sherpa.OfflineRecognizerConfig(
        nn_model=nn_model,
        tokens=tokens,
        use_gpu=False,
        feat_config=feat_config,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )

    recognizer = sherpa.OfflineRecognizer(config)

    return recognizer


@lru_cache(maxsize=10)
def _get_offline_pre_trained_model(
    repo_id: str, decoding_method: str, num_active_paths: int
) -> sherpa_onnx.OfflineRecognizer:
    assert repo_id in (
        "k2-fsa/sherpa-onnx-zipformer-korean-2024-06-24",
        "reazon-research/reazonspeech-k2-v2",
    ), repo_id

    encoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="encoder-epoch-99-avg-1.int8.onnx",
        subfolder=".",
    )

    decoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="decoder-epoch-99-avg-1.onnx",
        subfolder=".",
    )

    joiner_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="joiner-epoch-99-avg-1.onnx",
        subfolder=".",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder=".")

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder_model,
        decoder=decoder_model,
        joiner=joiner_model,
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=decoding_method,
    )

    return recognizer


@lru_cache(maxsize=10)
def _get_yifan_thai_pretrained_model(
    repo_id: str, decoding_method: str, num_active_paths: int
) -> sherpa_onnx.OfflineRecognizer:
    assert repo_id in (
        "yfyeung/icefall-asr-gigaspeech2-th-zipformer-2024-06-20",
    ), repo_id

    encoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="encoder-epoch-12-avg-5.int8.onnx",
        subfolder="exp",
    )

    decoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="decoder-epoch-12-avg-5.onnx",
        subfolder="exp",
    )

    joiner_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="joiner-epoch-12-avg-5.int8.onnx",
        subfolder="exp",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder="data/lang_bpe_2000")

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder_model,
        decoder=decoder_model,
        joiner=joiner_model,
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=decoding_method,
    )

    return recognizer


@lru_cache(maxsize=10)
def _get_zrjin_cantonese_pre_trained_model(
    repo_id: str, decoding_method: str, num_active_paths: int
) -> sherpa_onnx.OfflineRecognizer:
    assert repo_id in ("zrjin/icefall-asr-mdcc-zipformer-2024-03-11",), repo_id

    encoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="encoder-epoch-45-avg-35.int8.onnx",
        subfolder="exp",
    )

    decoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="decoder-epoch-45-avg-35.onnx",
        subfolder="exp",
    )

    joiner_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="joiner-epoch-45-avg-35.int8.onnx",
        subfolder="exp",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder="data/lang_char")

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder_model,
        decoder=decoder_model,
        joiner=joiner_model,
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=decoding_method,
    )

    return recognizer


@lru_cache(maxsize=10)
def _get_russian_pre_trained_model_ctc(
    repo_id: str, decoding_method: str, num_active_paths: int
) -> sherpa_onnx.OfflineRecognizer:
    assert repo_id in (
        "csukuangfj/sherpa-onnx-nemo-ctc-giga-am-russian-2024-10-24",
    ), repo_id

    model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="model.int8.onnx",
        subfolder=".",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder=".")

    recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=model,
        tokens=tokens,
        num_threads=2,
    )

    return recognizer


@lru_cache(maxsize=10)
def _get_russian_pre_trained_model(
    repo_id: str, decoding_method: str, num_active_paths: int
) -> sherpa_onnx.OfflineRecognizer:
    assert repo_id in (
        "alphacep/vosk-model-ru",
        "alphacep/vosk-model-small-ru",
        "csukuangfj/sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24",
    ), repo_id

    if repo_id == "alphacep/vosk-model-ru":
        model_dir = "am-onnx"
        encoder = "encoder.onnx"
        model_type = "transducer"
    elif repo_id == "alphacep/vosk-model-small-ru":
        model_dir = "am"
        encoder = "encoder.onnx"
        model_type = "transducer"
    elif repo_id == "csukuangfj/sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24":
        model_dir = "."
        encoder = "encoder.int8.onnx"
        model_type = "nemo_transducer"

    encoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename=encoder,
        subfolder=model_dir,
    )

    decoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="decoder.onnx",
        subfolder=model_dir,
    )

    joiner_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="joiner.onnx",
        subfolder=model_dir,
    )

    if repo_id == "csukuangfj/sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24":
        tokens = _get_token_filename(repo_id=repo_id, subfolder=".")
    else:
        tokens = _get_token_filename(repo_id=repo_id, subfolder="lang")

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder_model,
        decoder=decoder_model,
        joiner=joiner_model,
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=decoding_method,
        model_type=model_type,
    )

    return recognizer


@lru_cache(maxsize=10)
def _get_moonshine_model(
    repo_id: str, decoding_method: str, num_active_paths: int
) -> sherpa_onnx.OfflineRecognizer:
    assert repo_id in ("moonshine-tiny", "moonshine-base"), repo_id

    if repo_id == "moonshine-tiny":
        full_repo_id = "csukuangfj/sherpa-onnx-moonshine-tiny-en-int8"
    elif repo_id == "moonshine-base":
        full_repo_id = "csukuangfj/sherpa-onnx-moonshine-base-en-int8"
    else:
        raise ValueError(f"Unknown repo_id: {repo_id}")

    preprocessor = _get_nn_model_filename(
        repo_id=full_repo_id,
        filename=f"preprocess.onnx",
        subfolder=".",
    )

    encoder = _get_nn_model_filename(
        repo_id=full_repo_id,
        filename=f"encode.int8.onnx",
        subfolder=".",
    )

    uncached_decoder = _get_nn_model_filename(
        repo_id=full_repo_id,
        filename=f"uncached_decode.int8.onnx",
        subfolder=".",
    )

    cached_decoder = _get_nn_model_filename(
        repo_id=full_repo_id,
        filename=f"cached_decode.int8.onnx",
        subfolder=".",
    )

    tokens = _get_token_filename(
        repo_id=full_repo_id,
        subfolder=".",
        filename="tokens.txt",
    )

    recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine(
        preprocessor=preprocessor,
        encoder=encoder,
        uncached_decoder=uncached_decoder,
        cached_decoder=cached_decoder,
        tokens=tokens,
        num_threads=2,
    )

    return recognizer


@lru_cache(maxsize=10)
def _get_whisper_model(
    repo_id: str, decoding_method: str, num_active_paths: int
) -> sherpa_onnx.OfflineRecognizer:
    name = repo_id.split("-")[1]
    assert name in ("tiny.en", "base.en", "small.en", "medium.en"), repo_id
    full_repo_id = "csukuangfj/sherpa-onnx-whisper-" + name
    encoder = _get_nn_model_filename(
        repo_id=full_repo_id,
        filename=f"{name}-encoder.int8.onnx",
        subfolder=".",
    )

    decoder = _get_nn_model_filename(
        repo_id=full_repo_id,
        filename=f"{name}-decoder.int8.onnx",
        subfolder=".",
    )

    tokens = _get_token_filename(
        repo_id=full_repo_id, subfolder=".", filename=f"{name}-tokens.txt"
    )

    recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=encoder,
        decoder=decoder,
        tokens=tokens,
        num_threads=2,
    )

    return recognizer


@lru_cache(maxsize=10)
def _get_gigaspeech_pre_trained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
) -> sherpa.OfflineRecognizer:
    assert repo_id in [
        "wgb14/icefall-asr-gigaspeech-pruned-transducer-stateless2",
    ], repo_id

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="cpu_jit-iter-3488000-avg-20.pt",
    )
    tokens = "./giga-tokens.txt"

    feat_config = sherpa.FeatureConfig()
    feat_config.fbank_opts.frame_opts.samp_freq = sample_rate
    feat_config.fbank_opts.mel_opts.num_bins = 80
    feat_config.fbank_opts.frame_opts.dither = 0

    config = sherpa.OfflineRecognizerConfig(
        nn_model=nn_model,
        tokens=tokens,
        use_gpu=False,
        feat_config=feat_config,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )

    recognizer = sherpa.OfflineRecognizer(config)

    return recognizer


@lru_cache(maxsize=10)
def _get_english_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
) -> sherpa.OfflineRecognizer:
    assert repo_id in [
        "WeijiZhuang/icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02",  # noqa
        "yfyeung/icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04",  # noqa
        "yfyeung/icefall-asr-finetune-mux-pruned_transducer_stateless7-2023-05-19",  # noqa
        "csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13",  # noqa
        "csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11",  # noqa
        "csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless8-2022-11-14",  # noqa
        "Zengwei/icefall-asr-librispeech-zipformer-large-2023-05-16",  # noqa
        "Zengwei/icefall-asr-librispeech-zipformer-2023-05-15",  # noqa
        "Zengwei/icefall-asr-librispeech-zipformer-small-2023-05-16",  # noqa
        "videodanchik/icefall-asr-tedlium3-conformer-ctc2",
        "pkufool/icefall_asr_librispeech_conformer_ctc",
        "WayneWiser/icefall-asr-librispeech-conformer-ctc2-jit-bpe-500-2022-07-21",
    ], repo_id

    filename = "cpu_jit.pt"
    if (
        repo_id
        == "csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11"
    ):
        filename = "cpu_jit-torch-1.10.0.pt"

    if (
        repo_id
        == "WeijiZhuang/icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02"
    ):
        filename = "cpu_jit-torch-1.10.pt"

    if (
        repo_id
        == "yfyeung/icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04"
    ):
        filename = "cpu_jit-epoch-30-avg-4.pt"

    if (
        repo_id
        == "yfyeung/icefall-asr-finetune-mux-pruned_transducer_stateless7-2023-05-19"
    ):
        filename = "cpu_jit-epoch-20-avg-5.pt"

    if repo_id in (
        "Zengwei/icefall-asr-librispeech-zipformer-large-2023-05-16",
        "Zengwei/icefall-asr-librispeech-zipformer-2023-05-15",
        "Zengwei/icefall-asr-librispeech-zipformer-small-2023-05-16",
    ):
        filename = "jit_script.pt"

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename=filename,
    )
    subfolder = "data/lang_bpe_500"

    if repo_id in (
        "videodanchik/icefall-asr-tedlium3-conformer-ctc2",
        "pkufool/icefall_asr_librispeech_conformer_ctc",
    ):
        subfolder = "data/lang_bpe"

    tokens = _get_token_filename(repo_id=repo_id, subfolder=subfolder)

    feat_config = sherpa.FeatureConfig()
    feat_config.fbank_opts.frame_opts.samp_freq = sample_rate
    feat_config.fbank_opts.mel_opts.num_bins = 80
    feat_config.fbank_opts.frame_opts.dither = 0

    config = sherpa.OfflineRecognizerConfig(
        nn_model=nn_model,
        tokens=tokens,
        use_gpu=False,
        feat_config=feat_config,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )

    recognizer = sherpa.OfflineRecognizer(config)

    return recognizer


@lru_cache(maxsize=10)
def _get_wenetspeech_pre_trained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
):
    assert repo_id in [
        "luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless2",
    ], repo_id

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="cpu_jit_epoch_10_avg_2_torch_1.7.1.pt",
    )
    tokens = _get_token_filename(repo_id=repo_id)

    feat_config = sherpa.FeatureConfig()
    feat_config.fbank_opts.frame_opts.samp_freq = sample_rate
    feat_config.fbank_opts.mel_opts.num_bins = 80
    feat_config.fbank_opts.frame_opts.dither = 0

    config = sherpa.OfflineRecognizerConfig(
        nn_model=nn_model,
        tokens=tokens,
        use_gpu=False,
        feat_config=feat_config,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )

    recognizer = sherpa.OfflineRecognizer(config)

    return recognizer


@lru_cache(maxsize=10)
def _get_chinese_english_mixed_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
):
    assert repo_id in [
        "luomingshuang/icefall_asr_tal-csasr_pruned_transducer_stateless5",
        "ptrnull/icefall-asr-conv-emformer-transducer-stateless2-zh",
    ], repo_id

    if repo_id == "luomingshuang/icefall_asr_tal-csasr_pruned_transducer_stateless5":
        filename = "cpu_jit.pt"
        subfolder = "data/lang_char"
    elif repo_id == "ptrnull/icefall-asr-conv-emformer-transducer-stateless2-zh":
        filename = "cpu_jit-epoch-11-avg-1.pt"
        subfolder = "data/lang_char_bpe"

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename=filename,
    )
    tokens = _get_token_filename(repo_id=repo_id, subfolder=subfolder)

    feat_config = sherpa.FeatureConfig()
    feat_config.fbank_opts.frame_opts.samp_freq = sample_rate
    feat_config.fbank_opts.mel_opts.num_bins = 80
    feat_config.fbank_opts.frame_opts.dither = 0

    config = sherpa.OfflineRecognizerConfig(
        nn_model=nn_model,
        tokens=tokens,
        use_gpu=False,
        feat_config=feat_config,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )

    recognizer = sherpa.OfflineRecognizer(config)

    return recognizer


@lru_cache(maxsize=10)
def _get_alimeeting_pre_trained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
):
    assert repo_id in [
        "desh2608/icefall-asr-alimeeting-pruned-transducer-stateless7",
        "luomingshuang/icefall_asr_alimeeting_pruned_transducer_stateless2",
    ], repo_id

    if repo_id == "desh2608/icefall-asr-alimeeting-pruned-transducer-stateless7":
        filename = "cpu_jit.pt"
    elif repo_id == "luomingshuang/icefall_asr_alimeeting_pruned_transducer_stateless2":
        filename = "cpu_jit_torch_1.7.1.pt"

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename=filename,
    )
    tokens = _get_token_filename(repo_id=repo_id)

    feat_config = sherpa.FeatureConfig()
    feat_config.fbank_opts.frame_opts.samp_freq = sample_rate
    feat_config.fbank_opts.mel_opts.num_bins = 80
    feat_config.fbank_opts.frame_opts.dither = 0

    config = sherpa.OfflineRecognizerConfig(
        nn_model=nn_model,
        tokens=tokens,
        use_gpu=False,
        feat_config=feat_config,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )

    recognizer = sherpa.OfflineRecognizer(config)

    return recognizer


@lru_cache(maxsize=10)
def _get_wenet_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
):
    assert repo_id in [
        "csukuangfj/wenet-chinese-model",
        "csukuangfj/wenet-english-model",
    ], repo_id

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="final.zip",
        subfolder=".",
    )
    tokens = _get_token_filename(
        repo_id=repo_id,
        filename="units.txt",
        subfolder=".",
    )

    feat_config = sherpa.FeatureConfig(normalize_samples=False)
    feat_config.fbank_opts.frame_opts.samp_freq = sample_rate
    feat_config.fbank_opts.mel_opts.num_bins = 80
    feat_config.fbank_opts.frame_opts.dither = 0

    config = sherpa.OfflineRecognizerConfig(
        nn_model=nn_model,
        tokens=tokens,
        use_gpu=False,
        feat_config=feat_config,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )

    recognizer = sherpa.OfflineRecognizer(config)

    return recognizer


@lru_cache(maxsize=10)
def _get_aidatatang_200zh_pretrained_mode(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
):
    assert repo_id in [
        "luomingshuang/icefall_asr_aidatatang-200zh_pruned_transducer_stateless2",
    ], repo_id

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="cpu_jit_torch.1.7.1.pt",
    )
    tokens = _get_token_filename(repo_id=repo_id)

    feat_config = sherpa.FeatureConfig()
    feat_config.fbank_opts.frame_opts.samp_freq = sample_rate
    feat_config.fbank_opts.mel_opts.num_bins = 80
    feat_config.fbank_opts.frame_opts.dither = 0

    config = sherpa.OfflineRecognizerConfig(
        nn_model=nn_model,
        tokens=tokens,
        use_gpu=False,
        feat_config=feat_config,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )

    recognizer = sherpa.OfflineRecognizer(config)

    return recognizer


@lru_cache(maxsize=10)
def _get_tibetan_pre_trained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
):
    assert repo_id in [
        "syzym/icefall-asr-xbmu-amdo31-pruned-transducer-stateless7-2022-12-02",
        "syzym/icefall-asr-xbmu-amdo31-pruned-transducer-stateless5-2022-11-29",
    ], repo_id

    filename = "cpu_jit.pt"
    if (
        repo_id
        == "syzym/icefall-asr-xbmu-amdo31-pruned-transducer-stateless5-2022-11-29"
    ):
        filename = "cpu_jit-epoch-28-avg-23-torch-1.10.0.pt"

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename=filename,
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder="data/lang_bpe_500")

    feat_config = sherpa.FeatureConfig()
    feat_config.fbank_opts.frame_opts.samp_freq = sample_rate
    feat_config.fbank_opts.mel_opts.num_bins = 80
    feat_config.fbank_opts.frame_opts.dither = 0

    config = sherpa.OfflineRecognizerConfig(
        nn_model=nn_model,
        tokens=tokens,
        use_gpu=False,
        feat_config=feat_config,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )

    recognizer = sherpa.OfflineRecognizer(config)

    return recognizer


@lru_cache(maxsize=10)
def _get_arabic_pre_trained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
):
    assert repo_id in [
        "AmirHussein/icefall-asr-mgb2-conformer_ctc-2022-27-06",
    ], repo_id

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="cpu_jit.pt",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder="data/lang_bpe_5000")

    feat_config = sherpa.FeatureConfig()
    feat_config.fbank_opts.frame_opts.samp_freq = sample_rate
    feat_config.fbank_opts.mel_opts.num_bins = 80
    feat_config.fbank_opts.frame_opts.dither = 0

    config = sherpa.OfflineRecognizerConfig(
        nn_model=nn_model,
        tokens=tokens,
        use_gpu=False,
        feat_config=feat_config,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )

    recognizer = sherpa.OfflineRecognizer(config)

    return recognizer


@lru_cache(maxsize=10)
def _get_german_pre_trained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
):
    assert repo_id in [
        "csukuangfj/wav2vec2.0-torchaudio",
    ], repo_id

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="voxpopuli_asr_base_10k_de.pt",
        subfolder=".",
    )

    tokens = _get_token_filename(
        repo_id=repo_id,
        filename="tokens-de.txt",
        subfolder=".",
    )

    config = sherpa.OfflineRecognizerConfig(
        nn_model=nn_model,
        tokens=tokens,
        use_gpu=False,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )

    recognizer = sherpa.OfflineRecognizer(config)

    return recognizer


@lru_cache(maxsize=10)
def _get_french_pre_trained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
) -> sherpa_onnx.OnlineRecognizer:
    assert repo_id in [
        "shaojieli/sherpa-onnx-streaming-zipformer-fr-2023-04-14",
    ], repo_id

    encoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="encoder-epoch-29-avg-9-with-averaged-model.onnx",
        subfolder=".",
    )

    decoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="decoder-epoch-29-avg-9-with-averaged-model.onnx",
        subfolder=".",
    )

    joiner_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="joiner-epoch-29-avg-9-with-averaged-model.onnx",
        subfolder=".",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder=".")

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder_model,
        decoder=decoder_model,
        joiner=joiner_model,
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=decoding_method,
        max_active_paths=num_active_paths,
    )

    return recognizer


@lru_cache(maxsize=10)
def _get_sherpa_onnx_nemo_transducer_models(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
) -> sherpa_onnx.OfflineRecognizer:
    assert repo_id in [
        "csukuangfj/sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000",
    ], repo_id

    encoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="encoder.onnx",
        subfolder=".",
    )

    decoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="decoder.onnx",
        subfolder=".",
    )

    joiner_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="joiner.onnx",
        subfolder=".",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder=".")

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder_model,
        decoder=decoder_model,
        joiner=joiner_model,
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        model_type="nemo_transducer",
        decoding_method=decoding_method,
        max_active_paths=num_active_paths,
    )

    return recognizer


@lru_cache(maxsize=10)
def _get_sherpa_onnx_nemo_ctc_models(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
) -> sherpa_onnx.OfflineRecognizer:
    assert repo_id in [
        "csukuangfj/sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000",
    ], repo_id

    model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="model.onnx",
        subfolder=".",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder=".")

    recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        tokens=tokens,
        model=model,
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
    )

    return recognizer


@lru_cache(maxsize=10)
def _get_sherpa_onnx_offline_zipformer_pre_trained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
) -> sherpa_onnx.OfflineRecognizer:
    assert repo_id in [
        "csukuangfj/sherpa-onnx-zipformer-en-libriheavy-20230926-large",
        "csukuangfj/sherpa-onnx-zipformer-en-libriheavy-20230926-medium",
        "csukuangfj/sherpa-onnx-zipformer-en-libriheavy-20230926-small",
        "csukuangfj/sherpa-onnx-zipformer-en-libriheavy-20230830-large-punct-case",
        "csukuangfj/sherpa-onnx-zipformer-en-libriheavy-20230830-medium-punct-case",
        "csukuangfj/sherpa-onnx-zipformer-en-libriheavy-20230830-small-punct-case",
    ], repo_id

    if repo_id == "csukuangfj/sherpa-onnx-zipformer-en-libriheavy-20230926-large":
        epoch = 16
        avg = 3
    elif repo_id == "csukuangfj/sherpa-onnx-zipformer-en-libriheavy-20230926-medium":
        epoch = 60
        avg = 20
    elif repo_id == "csukuangfj/sherpa-onnx-zipformer-en-libriheavy-20230926-small":
        epoch = 90
        avg = 20
    elif (
        repo_id
        == "csukuangfj/sherpa-onnx-zipformer-en-libriheavy-20230830-large-punct-case"
    ):
        epoch = 16
        avg = 2
    elif (
        repo_id
        == "csukuangfj/sherpa-onnx-zipformer-en-libriheavy-20230830-medium-punct-case"
    ):
        epoch = 50
        avg = 15
    elif (
        repo_id
        == "csukuangfj/sherpa-onnx-zipformer-en-libriheavy-20230830-small-punct-case"
    ):
        epoch = 88
        avg = 41

    encoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename=f"encoder-epoch-{epoch}-avg-{avg}.int8.onnx",
        subfolder=".",
    )

    decoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename=f"decoder-epoch-{epoch}-avg-{avg}.onnx",
        subfolder=".",
    )

    joiner_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename=f"joiner-epoch-{epoch}-avg-{avg}.int8.onnx",
        subfolder=".",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder=".")

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder_model,
        decoder=decoder_model,
        joiner=joiner_model,
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=decoding_method,
        max_active_paths=num_active_paths,
    )

    return recognizer


@lru_cache(maxsize=10)
def _get_streaming_zipformer_pre_trained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
) -> sherpa_onnx.OnlineRecognizer:
    assert repo_id in [
        "csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
        "k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16",
    ], repo_id

    encoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="encoder-epoch-99-avg-1.onnx",
        subfolder=".",
    )

    decoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="decoder-epoch-99-avg-1.onnx",
        subfolder=".",
    )

    joiner_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="joiner-epoch-99-avg-1.onnx",
        subfolder=".",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder=".")

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder_model,
        decoder=decoder_model,
        joiner=joiner_model,
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=decoding_method,
        max_active_paths=num_active_paths,
    )

    return recognizer


@lru_cache(maxsize=10)
def _get_japanese_pre_trained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
) -> sherpa.OnlineRecognizer:
    repo_id, kind = repo_id.rsplit("-", maxsplit=1)

    assert repo_id in [
        "TeoWenShen/icefall-asr-csj-pruned-transducer-stateless7-streaming-230208"
    ], repo_id
    assert kind in ("fluent", "disfluent"), kind

    encoder_model = _get_nn_model_filename(
        repo_id=repo_id, filename="encoder_jit_trace.pt", subfolder=f"exp_{kind}"
    )

    decoder_model = _get_nn_model_filename(
        repo_id=repo_id, filename="decoder_jit_trace.pt", subfolder=f"exp_{kind}"
    )

    joiner_model = _get_nn_model_filename(
        repo_id=repo_id, filename="joiner_jit_trace.pt", subfolder=f"exp_{kind}"
    )

    tokens = _get_token_filename(repo_id=repo_id)

    feat_config = sherpa.FeatureConfig()
    feat_config.fbank_opts.frame_opts.samp_freq = sample_rate
    feat_config.fbank_opts.mel_opts.num_bins = 80
    feat_config.fbank_opts.frame_opts.dither = 0

    config = sherpa.OnlineRecognizerConfig(
        nn_model="",
        encoder_model=encoder_model,
        decoder_model=decoder_model,
        joiner_model=joiner_model,
        tokens=tokens,
        use_gpu=False,
        feat_config=feat_config,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
        chunk_size=32,
    )

    recognizer = sherpa.OnlineRecognizer(config)

    return recognizer


@lru_cache(maxsize=10)
def _get_gigaspeech_pre_trained_model_onnx(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
) -> sherpa_onnx.OfflineRecognizer:
    assert repo_id in [
        "yfyeung/icefall-asr-gigaspeech-zipformer-2023-10-17",
    ], repo_id

    encoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="encoder-epoch-30-avg-9.onnx",
        subfolder="exp",
    )

    decoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="decoder-epoch-30-avg-9.onnx",
        subfolder="exp",
    )

    joiner_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="joiner-epoch-30-avg-9.onnx",
        subfolder="exp",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder="data/lang_bpe_500")

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder_model,
        decoder=decoder_model,
        joiner=joiner_model,
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=decoding_method,
        max_active_paths=num_active_paths,
    )

    return recognizer


@lru_cache(maxsize=10)
def _get_streaming_paraformer_zh_yue_en_pre_trained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
) -> sherpa_onnx.OnlineRecognizer:
    assert repo_id in [
        "csukuangfj/sherpa-onnx-streaming-paraformer-trilingual-zh-cantonese-en",
    ], repo_id

    encoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="encoder.int8.onnx",
        subfolder=".",
    )

    decoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="decoder.int8.onnx",
        subfolder=".",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder=".")

    recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
        tokens=tokens,
        encoder=encoder_model,
        decoder=decoder_model,
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=decoding_method,
    )

    return recognizer


@lru_cache(maxsize=10)
def _get_paraformer_en_pre_trained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
) -> sherpa_onnx.OfflineRecognizer:
    assert repo_id in [
        "yujinqiu/sherpa-onnx-paraformer-en-2023-10-24",
    ], repo_id

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="model.int8.onnx",
        subfolder=".",
    )

    tokens = _get_token_filename(
        repo_id=repo_id, filename="new_tokens.txt", subfolder="."
    )

    recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=nn_model,
        tokens=tokens,
        num_threads=2,
        sample_rate=sample_rate,
        feature_dim=80,
        decoding_method="greedy_search",
        debug=False,
    )

    return recognizer


@lru_cache(maxsize=5)
def _get_chinese_dialect_models(
    repo_id: str, decoding_method: str, num_active_paths: int
) -> sherpa_onnx.OfflineRecognizer:
    assert repo_id in [
        "csukuangfj/sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04",
    ], repo_id

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="model.int8.onnx",
        subfolder=".",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder=".")

    recognizer = sherpa_onnx.OfflineRecognizer.from_telespeech_ctc(
        model=nn_model,
        tokens=tokens,
        num_threads=2,
    )

    return recognizer


@lru_cache(maxsize=10)
def _get_sense_voice_pre_trained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
) -> sherpa_onnx.OfflineRecognizer:
    assert repo_id in [
        "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
    ], repo_id

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="model.int8.onnx",
        subfolder=".",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder=".")

    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=nn_model,
        tokens=tokens,
        num_threads=2,
        sample_rate=sample_rate,
        feature_dim=80,
        decoding_method="greedy_search",
        debug=True,
        use_itn=True,
    )

    return recognizer


@lru_cache(maxsize=10)
def _get_paraformer_pre_trained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
) -> sherpa_onnx.OfflineRecognizer:
    assert repo_id in [
        "csukuangfj/sherpa-onnx-paraformer-zh-2023-03-28",
        "csukuangfj/sherpa-onnx-paraformer-zh-2024-03-09",
        "csukuangfj/sherpa-onnx-paraformer-zh-small-2024-03-09",
        "csukuangfj/sherpa-onnx-paraformer-trilingual-zh-cantonese-en",
        "csukuangfj/sherpa-onnx-paraformer-en-2024-03-09",
    ], repo_id

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="model.int8.onnx",
        subfolder=".",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder=".")

    recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=nn_model,
        tokens=tokens,
        num_threads=2,
        sample_rate=sample_rate,
        feature_dim=80,
        decoding_method="greedy_search",
        debug=False,
    )

    return recognizer


def _get_aishell_pre_trained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
) -> sherpa_onnx.OfflineRecognizer:
    assert repo_id in (
        "zrjin/icefall-asr-aishell-zipformer-large-2023-10-24",
        "zrjin/icefall-asr-aishell-zipformer-small-2023-10-24",
        "zrjin/icefall-asr-aishell-zipformer-2023-10-24",
    ), repo_id
    if repo_id == "zrjin/icefall-asr-aishell-zipformer-large-2023-10-24":
        epoch = 56
        avg = 23
    elif repo_id == "zrjin/icefall-asr-aishell-zipformer-small-2023-10-24":
        epoch = 55
        avg = 21
    elif repo_id == "zrjin/icefall-asr-aishell-zipformer-2023-10-24":
        epoch = 55
        avg = 17

    encoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename=f"encoder-epoch-{epoch}-avg-{avg}.onnx",
        subfolder="exp",
    )

    decoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename=f"decoder-epoch-{epoch}-avg-{avg}.onnx",
        subfolder="exp",
    )

    joiner_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename=f"joiner-epoch-{epoch}-avg-{avg}.onnx",
        subfolder="exp",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder="data/lang_char")

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder_model,
        decoder=decoder_model,
        joiner=joiner_model,
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=decoding_method,
        max_active_paths=num_active_paths,
    )

    return recognizer


@lru_cache(maxsize=2)
def get_punct_model() -> sherpa_onnx.OfflinePunctuation:
    model = _get_nn_model_filename(
        repo_id="csukuangfj/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12",
        filename="model.onnx",
        subfolder=".",
    )
    config = sherpa_onnx.OfflinePunctuationConfig(
        model=sherpa_onnx.OfflinePunctuationModelConfig(ct_transformer=model),
    )

    punct = sherpa_onnx.OfflinePunctuation(config)
    return punct


def _get_multi_zh_hans_pre_trained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
) -> sherpa_onnx.OfflineRecognizer:
    assert repo_id in ("zrjin/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2",), repo_id

    encoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="encoder-epoch-20-avg-1.onnx",
        subfolder=".",
    )

    decoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="decoder-epoch-20-avg-1.onnx",
        subfolder=".",
    )

    joiner_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="joiner-epoch-20-avg-1.onnx",
        subfolder=".",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder=".")

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder_model,
        decoder=decoder_model,
        joiner=joiner_model,
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=decoding_method,
        max_active_paths=num_active_paths,
    )

    return recognizer


chinese_dialect_models = {
    "csukuangfj/sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04": _get_chinese_dialect_models,
}

chinese_models = {
    "csukuangfj/sherpa-onnx-paraformer-zh-2024-03-09": _get_paraformer_pre_trained_model,
    "luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless2": _get_wenetspeech_pre_trained_model,  # noqa
    "csukuangfj/sherpa-onnx-paraformer-zh-small-2024-03-09": _get_paraformer_pre_trained_model,
    "zrjin/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2": _get_multi_zh_hans_pre_trained_model,  # noqa
    "zrjin/icefall-asr-aishell-zipformer-large-2023-10-24": _get_aishell_pre_trained_model,  # noqa
    "zrjin/icefall-asr-aishell-zipformer-small-2023-10-24": _get_aishell_pre_trained_model,  # noqa
    "zrjin/icefall-asr-aishell-zipformer-2023-10-24": _get_aishell_pre_trained_model,  # noqa
    "desh2608/icefall-asr-alimeeting-pruned-transducer-stateless7": _get_alimeeting_pre_trained_model,
    "yuekai/icefall-asr-aishell2-pruned-transducer-stateless5-A-2022-07-12": _get_aishell2_pretrained_model,  # noqa
    "yuekai/icefall-asr-aishell2-pruned-transducer-stateless5-B-2022-07-12": _get_aishell2_pretrained_model,  # noqa
    "luomingshuang/icefall_asr_aidatatang-200zh_pruned_transducer_stateless2": _get_aidatatang_200zh_pretrained_mode,  # noqa
    "luomingshuang/icefall_asr_alimeeting_pruned_transducer_stateless2": _get_alimeeting_pre_trained_model,  # noqa
    "csukuangfj/wenet-chinese-model": _get_wenet_model,
    #  "csukuangfj/icefall-asr-wenetspeech-lstm-transducer-stateless-2022-10-14": _get_lstm_transducer_model,
}

english_models = {
    "whisper-tiny.en": _get_whisper_model,
    "moonshine-tiny": _get_moonshine_model,
    "moonshine-base": _get_moonshine_model,
    "whisper-base.en": _get_whisper_model,
    "whisper-small.en": _get_whisper_model,
    "csukuangfj/sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000": _get_sherpa_onnx_nemo_ctc_models,
    "csukuangfj/sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000": _get_sherpa_onnx_nemo_transducer_models,
    #  "whisper-medium.en": _get_whisper_model,
    "csukuangfj/sherpa-onnx-zipformer-en-libriheavy-20230926-large": _get_sherpa_onnx_offline_zipformer_pre_trained_model,
    "csukuangfj/sherpa-onnx-zipformer-en-libriheavy-20230926-medium": _get_sherpa_onnx_offline_zipformer_pre_trained_model,
    "csukuangfj/sherpa-onnx-zipformer-en-libriheavy-20230926-small": _get_sherpa_onnx_offline_zipformer_pre_trained_model,
    "csukuangfj/sherpa-onnx-zipformer-en-libriheavy-20230830-large-punct-case": _get_sherpa_onnx_offline_zipformer_pre_trained_model,
    "csukuangfj/sherpa-onnx-zipformer-en-libriheavy-20230830-medium-punct-case": _get_sherpa_onnx_offline_zipformer_pre_trained_model,
    "csukuangfj/sherpa-onnx-zipformer-en-libriheavy-20230830-small-punct-case": _get_sherpa_onnx_offline_zipformer_pre_trained_model,
    "csukuangfj/sherpa-onnx-paraformer-en-2024-03-09": _get_paraformer_pre_trained_model,
    "yfyeung/icefall-asr-gigaspeech-zipformer-2023-10-17": _get_gigaspeech_pre_trained_model_onnx,  # noqa
    "wgb14/icefall-asr-gigaspeech-pruned-transducer-stateless2": _get_gigaspeech_pre_trained_model,  # noqa
    "yfyeung/icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04": _get_english_model,  # noqa
    "yfyeung/icefall-asr-finetune-mux-pruned_transducer_stateless7-2023-05-19": _get_english_model,  # noqa
    "WeijiZhuang/icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02": _get_english_model,  # noqa
    "csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless8-2022-11-14": _get_english_model,  # noqa
    "csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11": _get_english_model,  # noqa
    "csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13": _get_english_model,  # noqa
    "yujinqiu/sherpa-onnx-paraformer-en-2023-10-24": _get_paraformer_en_pre_trained_model,
    "Zengwei/icefall-asr-librispeech-zipformer-large-2023-05-16": _get_english_model,  # noqa
    "Zengwei/icefall-asr-librispeech-zipformer-2023-05-15": _get_english_model,  # noqa
    "Zengwei/icefall-asr-librispeech-zipformer-small-2023-05-16": _get_english_model,  # noqa
    "videodanchik/icefall-asr-tedlium3-conformer-ctc2": _get_english_model,
    "pkufool/icefall_asr_librispeech_conformer_ctc": _get_english_model,
    "WayneWiser/icefall-asr-librispeech-conformer-ctc2-jit-bpe-500-2022-07-21": _get_english_model,
    "csukuangfj/wenet-english-model": _get_wenet_model,
}

chinese_english_mixed_models = {
    "csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20": _get_streaming_zipformer_pre_trained_model,
    "csukuangfj/sherpa-onnx-paraformer-zh-2023-03-28": _get_paraformer_pre_trained_model,
    "ptrnull/icefall-asr-conv-emformer-transducer-stateless2-zh": _get_chinese_english_mixed_model,
    "luomingshuang/icefall_asr_tal-csasr_pruned_transducer_stateless5": _get_chinese_english_mixed_model,  # noqa
}

tibetan_models = {
    "syzym/icefall-asr-xbmu-amdo31-pruned-transducer-stateless7-2022-12-02": _get_tibetan_pre_trained_model,  # noqa
    "syzym/icefall-asr-xbmu-amdo31-pruned-transducer-stateless5-2022-11-29": _get_tibetan_pre_trained_model,  # noqa
}

arabic_models = {
    "AmirHussein/icefall-asr-mgb2-conformer_ctc-2022-27-06": _get_arabic_pre_trained_model,  # noqa
}

german_models = {
    "csukuangfj/wav2vec2.0-torchaudio": _get_german_pre_trained_model,
}

french_models = {
    "shaojieli/sherpa-onnx-streaming-zipformer-fr-2023-04-14": _get_french_pre_trained_model,
}

japanese_models = {
    "reazon-research/reazonspeech-k2-v2": _get_offline_pre_trained_model,
    #  "TeoWenShen/icefall-asr-csj-pruned-transducer-stateless7-streaming-230208-fluent": _get_japanese_pre_trained_model,
    #  "TeoWenShen/icefall-asr-csj-pruned-transducer-stateless7-streaming-230208-disfluent": _get_japanese_pre_trained_model,
}

russian_models = {
    "csukuangfj/sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24": _get_russian_pre_trained_model,
    "csukuangfj/sherpa-onnx-nemo-ctc-giga-am-russian-2024-10-24": _get_russian_pre_trained_model_ctc,
    "alphacep/vosk-model-ru": _get_russian_pre_trained_model,
    "alphacep/vosk-model-small-ru": _get_russian_pre_trained_model,
}

chinese_cantonese_english_models = {
    "csukuangfj/sherpa-onnx-paraformer-trilingual-zh-cantonese-en": _get_paraformer_pre_trained_model,
    "csukuangfj/sherpa-onnx-streaming-paraformer-trilingual-zh-cantonese-en": _get_streaming_paraformer_zh_yue_en_pre_trained_model,
}

chinese_cantonese_english_japanese_korean_models = {
    "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17": _get_sense_voice_pre_trained_model,
}

cantonese_models = {
    "zrjin/icefall-asr-mdcc-zipformer-2024-03-11": _get_zrjin_cantonese_pre_trained_model,
}

korean_models = {
    "k2-fsa/sherpa-onnx-zipformer-korean-2024-06-24": _get_offline_pre_trained_model,
    "k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16": _get_streaming_zipformer_pre_trained_model,
}

thai_models = {
    "yfyeung/icefall-asr-gigaspeech2-th-zipformer-2024-06-20": _get_yifan_thai_pretrained_model,
}


all_models = {
    **chinese_models,
    **english_models,
    **chinese_english_mixed_models,
    **chinese_cantonese_english_models,
    **chinese_cantonese_english_japanese_korean_models,
    **cantonese_models,
    **japanese_models,
    **tibetan_models,
    **arabic_models,
    **german_models,
    **french_models,
    **russian_models,
    **korean_models,
    **thai_models,
}

language_to_models = {
    "超多种中文方言": list(chinese_dialect_models.keys()),
    "Chinese": list(chinese_models.keys()),
    "English": list(english_models.keys()),
    "Chinese+English": list(chinese_english_mixed_models.keys()),
    "Chinese+English+Cantonese": list(chinese_cantonese_english_models.keys()),
    "Chinese+English+Cantonese+Japanese+Korean": list(
        chinese_cantonese_english_japanese_korean_models.keys()
    ),
    "Cantonese": list(cantonese_models.keys()),
    "Japanese": list(japanese_models.keys()),
    "Tibetan": list(tibetan_models.keys()),
    "Arabic": list(arabic_models.keys()),
    "German": list(german_models.keys()),
    "French": list(french_models.keys()),
    "Russian": list(russian_models.keys()),
    "Korean": list(korean_models.keys()),
    "Thai": list(thai_models.keys()),
}
