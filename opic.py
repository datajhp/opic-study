import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import numpy as np
import queue

st.title("🎙️ Streamlit WebRTC 진단용 녹음기")

# 오디오 큐 생성
if "audio_queue" not in st.session_state:
    st.session_state.audio_queue = queue.Queue()

# recv() 호출 로그 저장
if "recv_called" not in st.session_state:
    st.session_state.recv_called = False

# Audio Processor 정의
class AudioProcessor(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray().flatten().astype(np.float32)
        st.session_state.audio_queue.put(pcm)
        st.session_state.recv_called = True
        return frame

# WebRTC 연결
st.header("🔌 마이크 연결 상태 확인")
webrtc_ctx = webrtc_streamer(
    key="debug-mic",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor,
    async_processing=True,
)

# 연결 상태 출력
if webrtc_ctx:
    st.write("🎯 WebRTC 상태:", webrtc_ctx.state)
    if webrtc_ctx.state.playing:
        st.success("✅ playing = True (마이크 스트림 수신 중)")
    elif webrtc_ctx.state.connected:
        st.warning("🟡 connected = True (아직 재생 안 됨)")
    else:
        st.error("❌ WebRTC 연결 실패")

# recv() 호출 여부
st.header("📡 프레임 수신 상태")
st.write("✅ recv() 호출됨:", st.session_state.recv_called)

# 오디오 큐 길이 출력
st.write("🎧 audio_queue 길이:", len(st.session_state.audio_queue.queue))

# 종료 후 분석 버튼
if st.button("🔍 분석 테스트 (녹음 종료 후 실행 가정)"):
    if len(st.session_state.audio_queue.queue) > 0:
        st.success("🎉 녹음 데이터 수신 성공! Whisper 분석 가능")
    else:
        st.error("❌ 아직 녹음된 오디오가 없습니다. 마이크 권한/입력 확인 필요")

# 리셋 기능
if st.button("♻️ 상태 초기화"):
    st.session_state.audio_queue = queue.Queue()
    st.session_state.recv_called = False
    st.success("상태 초기화 완료")
