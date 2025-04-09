import streamlit as st
import whisper
import tempfile
import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import queue

# 모델 로딩
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

@st.cache_resource
def load_grammar_model():
    tokenizer = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction")
    model = AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction")
    return tokenizer, model

# 문법 피드백 함수
def grammar_correction(text):
    tokenizer, model = load_grammar_model()
    inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs, max_length=512, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

st.title("🎤 오픽 영어 말하기 연습 앱")

tabs = st.tabs(["📁 파일 업로드", "🎙 마이크 녹음", "✍️ 텍스트 입력"])

# 1. 텍스트 입력 탭
with tabs[2]:
    user_input = st.text_area("오픽 응답을 영어로 직접 입력하세요:", height=150)
    if st.button("✅ 문법 피드백 받기"):
        if user_input.strip() == "":
            st.warning("문장을 입력해주세요.")
        else:
            st.info("문법 교정 중입니다...")
            corrected = grammar_correction(user_input)
            st.subheader("✅ 교정된 문장")
            st.success(corrected)

# 2. 음성 파일 업로드 탭
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

# 3. 마이크 녹음 탭
with tabs[1]:
    st.title("🎙 오픽 말하기 녹음 연습")

# 1단계: 녹음 시작
if not st.session_state.recording:
    if st.button("🎤 녹음 시작"):
        st.session_state.recording = True
        st.experimental_rerun()
else:
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


