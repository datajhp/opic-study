import streamlit as st
import whisper
import tempfile
import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import queue

# 모델 로딩 (캐시 처리)
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

@st.cache_resource
def load_grammar_model():
    tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
    model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")
    return tokenizer, model

# 문법 피드백 함수
def grammar_correction(text):
    tokenizer, model = load_grammar_model()
    inputs = tokenizer.encode("gec: " + text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs, max_length=512, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

st.title("🎤 오픽 영어 말하기 연습 (파일 업로드 + 마이크 지원)")

tabs = st.tabs(["📁 파일 업로드", "🎙 마이크 녹음"])

# 음성 파일 업로드 방식
with tabs[0]:
    uploaded = st.file_uploader("오픽 응답 음성 파일 업로드 (.wav / .mp3)", type=["wav", "mp3"])
    if uploaded:
        st.audio(uploaded)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.info("음성을 텍스트로 변환 중입니다...")
        whisper_model = load_whisper()
        result = whisper_model.transcribe(tmp_path)
        user_text = result["text"]
        st.markdown("**🎧 전사 결과:**")
        st.write(user_text)

        corrected = grammar_correction(user_text)
        st.markdown("**✅ 교정된 문장:**")
        st.success(corrected)

# 마이크 녹음 방식
with tabs[1]:
    st.info("🎤 브라우저에서 마이크 권한을 허용해주세요!")
    audio_queue = queue.Queue()

    class AudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.recorded_frames = []

        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            pcm = frame.to_ndarray().flatten().astype(np.float32)
            audio_queue.put(pcm)
            return frame

    webrtc_ctx = webrtc_streamer(
        key="mic",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        media_stream_constraints={"audio": True, "video": False},
        audio_processor_factory=AudioProcessor,
        async_processing=True,
    )

    if st.button("✅ 녹음 종료하고 분석하기"):
        if not audio_queue.empty():
            st.info("🔄 음성 분석 중입니다...")

            all_audio = []
            while not audio_queue.empty():
                all_audio.extend(audio_queue.get())
            audio_tensor = torch.tensor(all_audio)
            sample_rate = 48000  # 기본 스트리밍 샘플링

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_audio_path = f.name
                import torchaudio
                torchaudio.save(tmp_audio_path, audio_tensor.unsqueeze(0), sample_rate)

            whisper_model = load_whisper()
            result = whisper_model.transcribe(tmp_audio_path)
            user_text = result["text"]
            st.markdown("**🎧 전사 결과:**")
            st.write(user_text)

            corrected = grammar_correction(user_text)
            st.markdown("**✅ 교정된 문장:**")
            st.success(corrected)
        else:
            st.warning("녹음된 음성이 없습니다. 마이크 권한과 상태를 확인해주세요.")
