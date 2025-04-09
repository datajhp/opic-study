import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import whisper
import torchaudio
import torch
import tempfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import av
import numpy as np
import queue

# Whisper + grammar model 로딩
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

@st.cache_resource
def load_grammar_model():
    tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
    model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")
    return tokenizer, model

st.title("🎙 마이크로 오픽 응답 녹음 + 피드백 시스템")

# 오디오 버퍼용 큐 생성
audio_queue = queue.Queue()

# 오디오 프로세서 클래스 정의
class AudioProcessor(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Float32로 변환
        pcm = frame.to_ndarray().flatten().astype(np.float32)
        audio_queue.put(pcm)
        return frame

# 마이크 스트리밍 시작
webrtc_ctx = webrtc_streamer(
    key="mic",
    mode=WebRtcMode.SENDONLY,  # ✅ 올바른 enum 타입으로 설정
    audio_receiver_size=1024,
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor,
    async_processing=True,
)

# 녹음이 완료되면 버튼 클릭
if st.button("✅ 녹음 종료 후 피드백 받기"):
    st.info("녹음된 음성을 저장하고 분석 중입니다...")

    # 오디오 데이터 가져오기
    all_audio = []
    while not audio_queue.empty():
        all_audio.extend(audio_queue.get())

    audio_tensor = torch.tensor(all_audio)
    sample_rate = 48000  # streamlit_webrtc 기본 샘플링

    # 임시 wav 파일로 저장
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        torchaudio.save(f.name, audio_tensor.unsqueeze(0), sample_rate)
        audio_path = f.name

    # Whisper로 텍스트 전환
    whisper_model = load_whisper()
    result = whisper_model.transcribe(audio_path)
    transcript = result["text"]

    st.subheader("📝 전사된 텍스트")
    st.write(transcript)

    # 문법 교정
    tokenizer, grammar_model = load_grammar_model()
    input_ids = tokenizer.encode("gec: " + transcript, return_tensors="pt", max_length=512, truncation=True)
    outputs = grammar_model.generate(input_ids, max_length=512, num_beams=4)
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.subheader("✅ 교정 결과")
    st.markdown(f"**수정된 문장:** {corrected}")

