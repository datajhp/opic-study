import streamlit as st
import whisper
import tempfile
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import queue

# 전역 변수로 녹음 상태를 관리
st.session_state.setdefault("recording", False)

# 오디오 버퍼 큐
audio_queue = queue.Queue()

# Audio 처리 클래스
class AudioProcessor(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray().flatten().astype(np.float32)
        audio_queue.put(pcm)
        return frame

# 모델 로딩
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

@st.cache_resource
def load_grammar_model():
    tokenizer = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction")
    model = AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction")
    return tokenizer, model

def grammar_correction(text):
    tokenizer, model = load_grammar_model()
    inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs, max_length=512, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

st.title("🎙 오픽 말하기 녹음 연습")

if "recording" not in st.session_state:
    st.session_state.recording = False
# 1단계: 녹음 시작
if not st.session_state.recording:
    if st.button("🎤 녹음 시작"):
        st.session_state.recording = True
        # ❌ st.experimental_rerun() 제거
if st.session_state.recording:
    st.success("🔴 녹음 중입니다! 말하고 나서 아래 버튼을 눌러주세요.")

    webrtc_ctx = webrtc_streamer(
        key="mic-recorder",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        media_stream_constraints={"audio": True, "video": False},
        audio_processor_factory=AudioProcessor,
        async_processing=True,
    )

    if webrtc_ctx and webrtc_ctx.state.playing:
        st.info("🎧 마이크가 연결되었습니다. 말해보세요!")
    elif webrtc_ctx and not webrtc_ctx.state.playing:
        st.warning("⏳ 마이크 연결 대기 중입니다...")
    else:
        st.warning("🛑 마이크 초기화 중입니다...")

    # 2단계: 녹음 종료 + 분석
    if st.button("✅ 녹음 종료 및 분석"):
        st.session_state.recording = False

        if not audio_queue.empty():
            all_audio = []
            while not audio_queue.empty():
                all_audio.extend(audio_queue.get())
            audio_tensor = torch.tensor(all_audio)
            sample_rate = 48000

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_audio_path = f.name
                import torchaudio
                torchaudio.save(tmp_audio_path, audio_tensor.unsqueeze(0), sample_rate)

            whisper_model = load_whisper()
            result = whisper_model.transcribe(tmp_audio_path)
            user_text = result["text"]
            st.subheader("📝 전사된 텍스트")
            st.write(user_text)

            corrected = grammar_correction(user_text)
            st.subheader("✅ 교정된 문장")
            st.success(corrected)
        else:
            st.error("🎙 녹음된 음성이 없습니다. 마이크 권한 또는 연결 상태를 확인해주세요.")
