#!/usr/bin/env python3
#
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

# References:
# https://gradio.app/docs/#dropdown

import logging
import os
import tempfile
import time
import urllib.request
import uuid
from datetime import datetime

import gradio as gr
import torch
import torchaudio

from examples import examples
from model import (
    decode,
    get_pretrained_model,
    get_punct_model,
    language_to_models,
    sample_rate,
)

languages = list(language_to_models.keys())


def MyPrint(s):
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    print(f"{date_time}: {s}")


def convert_to_wav(in_filename: str) -> str:
    """Convert the input audio file to a wave file"""
    out_filename = str(uuid.uuid4())
    out_filename = f"{in_filename}.wav"

    MyPrint(f"Converting '{in_filename}' to '{out_filename}'")
    _ = os.system(
        f"ffmpeg -hide_banner -loglevel error -i '{in_filename}' -ar 16000 -ac 1 '{out_filename}' -y"
    )

    return out_filename


def build_html_output(s: str, style: str = "result_item_success"):
    return f"""
    <div class='result'>
        <div class='result_item {style}'>
          {s}
        </div>
    </div>
    """


def process_url(
    language: str,
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
    add_punct: str,
    url: str,
):
    MyPrint(f"Processing URL: {url}")
    with tempfile.NamedTemporaryFile() as f:
        try:
            urllib.request.urlretrieve(url, f.name)

            return process(
                in_filename=f.name,
                language=language,
                repo_id=repo_id,
                decoding_method=decoding_method,
                num_active_paths=num_active_paths,
                add_punct=add_punct,
            )
        except Exception as e:
            MyPrint(str(e))
            return "", build_html_output(str(e), "result_item_error")


def process_uploaded_file(
    language: str,
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
    add_punct: str,
    in_filename: str,
):
    if in_filename is None or in_filename == "":
        return "", build_html_output(
            "Please first upload a file and then click "
            'the button "submit for recognition"',
            "result_item_error",
        )

    MyPrint(f"Processing uploaded file: {in_filename}")
    try:
        return process(
            in_filename=in_filename,
            language=language,
            repo_id=repo_id,
            decoding_method=decoding_method,
            num_active_paths=num_active_paths,
            add_punct=add_punct,
        )
    except Exception as e:
        MyPrint(str(e))
        return "", build_html_output(str(e), "result_item_error")


def process_microphone(
    language: str,
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
    add_punct: str,
    in_filename: str,
):
    if in_filename is None or in_filename == "":
        return "", build_html_output(
            "Please first click 'Record from microphone', speak, "
            "click 'Stop recording', and then "
            "click the button 'submit for recognition'",
            "result_item_error",
        )

    MyPrint(f"Processing microphone: {in_filename}")
    try:
        return process(
            in_filename=in_filename,
            language=language,
            repo_id=repo_id,
            decoding_method=decoding_method,
            num_active_paths=num_active_paths,
            add_punct=add_punct,
        )
    except Exception as e:
        MyPrint(str(e))
        return "", build_html_output(str(e), "result_item_error")


@torch.no_grad()
def process(
    language: str,
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
    add_punct: str,
    in_filename: str,
):
    MyPrint(f"language: {language}")
    MyPrint(f"repo_id: {repo_id}")
    MyPrint(f"decoding_method: {decoding_method}")
    MyPrint(f"num_active_paths: {num_active_paths}")
    MyPrint(f"in_filename: {in_filename}")

    filename = convert_to_wav(in_filename)

    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    MyPrint(f"Started at {date_time}")

    start = time.time()

    recognizer = get_pretrained_model(
        repo_id,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )

    text = decode(recognizer, filename)
    if add_punct == "Yes" and language == "Chinese":
        punct = get_punct_model()
        text = punct.add_punctuation(text)

    date_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    end = time.time()

    metadata = torchaudio.info(filename)
    duration = metadata.num_frames / sample_rate
    rtf = (end - start) / duration

    MyPrint(f"Finished at {date_time} s. Elapsed: {end - start: .3f} s")

    info = f"""
    Wave duration  : {duration: .3f} s <br/>
    Processing time: {end - start: .3f} s <br/>
    RTF: {end - start: .3f}/{duration: .3f} = {rtf:.3f} <br/>
    """
    if rtf > 1:
        info += (
            "<br/>We are loading the model for the first run. "
            "Please run again to measure the real RTF.<br/>"
        )

    MyPrint(info)
    MyPrint(f"\nrepo_id: {repo_id}\nhyp: {text}")

    return text, build_html_output(info)


title = "# Automatic Speech Recognition with Next-gen Kaldi"
description = """
This space shows how to do automatic speech recognition with Next-gen Kaldi.

Please visit
<https://k2-fsa.github.io/sherpa/ncnn/wasm/hf-spaces.html>
for streaming speech recognition with **Next-gen Kaldi** using WebAssembly.

It is running on CPU within a docker container provided by Hugging Face.

Please input audio files less than 30 seconds in this space.

Please see <https://huggingface.co/spaces/k2-fsa/generate-subtitles-for-videos>
if you want to try files longer than 30 seconds.

For text to speech, please see
<https://huggingface.co/spaces/k2-fsa/text-to-speech>

See more information by visiting the following links:

- <https://github.com/k2-fsa/icefall>
- <https://github.com/k2-fsa/sherpa>
- <https://github.com/k2-fsa/sherpa-onnx>
- <https://github.com/k2-fsa/sherpa-ncnn>
- <https://github.com/k2-fsa/k2>
- <https://github.com/lhotse-speech/lhotse>

If you want to deploy it locally, please see
<https://k2-fsa.github.io/sherpa/>
"""

# css style is copied from
# https://huggingface.co/spaces/alphacep/asr/blob/main/app.py#L113
css = """
.result {display:flex;flex-direction:column}
.result_item {padding:15px;margin-bottom:8px;border-radius:15px;width:100%}
.result_item_success {background-color:mediumaquamarine;color:white;align-self:start}
.result_item_error {background-color:#ff7070;color:white;align-self:start}
"""


def update_model_dropdown(language: str):
    if language in language_to_models:
        choices = language_to_models[language]
        return gr.Dropdown(
            choices=choices,
            value=choices[0],
            interactive=True,
        )

    raise ValueError(f"Unsupported language: {language}")


demo = gr.Blocks(css=css)


with demo:
    gr.Markdown(title)
    language_choices = list(language_to_models.keys())

    language_radio = gr.Radio(
        label="Language",
        choices=language_choices,
        value=language_choices[0],
    )
    model_dropdown = gr.Dropdown(
        choices=language_to_models[language_choices[0]],
        label="Select a model",
        value=language_to_models[language_choices[0]][0],
    )

    language_radio.change(
        update_model_dropdown,
        inputs=language_radio,
        outputs=model_dropdown,
    )

    decoding_method_radio = gr.Radio(
        label="Decoding method",
        choices=["greedy_search", "modified_beam_search"],
        value="greedy_search",
    )

    num_active_paths_slider = gr.Slider(
        minimum=1,
        value=4,
        step=1,
        label="Number of active paths for modified_beam_search",
    )

    punct_radio = gr.Radio(
        label="Whether to add punctuation (Only for Chinese)",
        choices=["Yes", "No"],
        value="Yes",
    )

    with gr.Tabs():
        with gr.TabItem("Upload from disk"):
            uploaded_file = gr.Audio(
                sources=["upload"],  # Choose between "microphone", "upload"
                type="filepath",
                label="Upload from disk",
            )
            upload_button = gr.Button("Submit for recognition")
            uploaded_output = gr.Textbox(label="Recognized speech from uploaded file")
            uploaded_html_info = gr.HTML(label="Info")

            gr.Examples(
                examples=examples,
                inputs=[
                    language_radio,
                    model_dropdown,
                    decoding_method_radio,
                    num_active_paths_slider,
                    punct_radio,
                    uploaded_file,
                ],
                outputs=[uploaded_output, uploaded_html_info],
                fn=process_uploaded_file,
            )

        with gr.TabItem("Record from microphone"):
            microphone = gr.Audio(
                sources=["microphone"],  # Choose between "microphone", "upload"
                type="filepath",
                label="Record from microphone",
            )

            record_button = gr.Button("Submit for recognition")
            recorded_output = gr.Textbox(label="Recognized speech from recordings")
            recorded_html_info = gr.HTML(label="Info")

            gr.Examples(
                examples=examples,
                inputs=[
                    language_radio,
                    model_dropdown,
                    decoding_method_radio,
                    num_active_paths_slider,
                    punct_radio,
                    microphone,
                ],
                outputs=[recorded_output, recorded_html_info],
                fn=process_microphone,
            )

        with gr.TabItem("From URL"):
            url_textbox = gr.Textbox(
                max_lines=1,
                placeholder="URL to an audio file",
                label="URL",
                interactive=True,
            )

            url_button = gr.Button("Submit for recognition")
            url_output = gr.Textbox(label="Recognized speech from URL")
            url_html_info = gr.HTML(label="Info")

        upload_button.click(
            process_uploaded_file,
            inputs=[
                language_radio,
                model_dropdown,
                decoding_method_radio,
                num_active_paths_slider,
                punct_radio,
                uploaded_file,
            ],
            outputs=[uploaded_output, uploaded_html_info],
        )

        record_button.click(
            process_microphone,
            inputs=[
                language_radio,
                model_dropdown,
                decoding_method_radio,
                num_active_paths_slider,
                punct_radio,
                microphone,
            ],
            outputs=[recorded_output, recorded_html_info],
        )

        url_button.click(
            process_url,
            inputs=[
                language_radio,
                model_dropdown,
                decoding_method_radio,
                num_active_paths_slider,
                punct_radio,
                url_textbox,
            ],
            outputs=[url_output, url_html_info],
        )

    gr.Markdown(description)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)

if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    demo.launch()
