import streamlit as st
from audiorecorder import audiorecorder
from speechbrain.inference import EncoderDecoderASR

st.title('SpeechBrain - Speech to Text')

audio = audiorecorder("Record")

def convertspeechtotext():
    try:
        # Load pretrained ASR model
        asr_model = EncoderDecoderASR.from_hparams(
            source="speechbrain/asr-conformer-transformerlm-librispeech",
            savedir="pretrained_models/asr-transformer-transformerlm-librispeech"
        )

        # Transcribe the audio file
        text = asr_model.transcribe_file("audio.wav")
        return text
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

if len(audio) > 0:
    # Export the audio and write it to a file
    audio_file_path = "audio.wav"
    with open(audio_file_path, "wb") as f:
        f.write(audio.export().read())

    # Display the audio player
    st.audio(audio.export().read(), format="audio/wav", autoplay=True)

    # Convert speech to text
    Transcript = convertspeechtotext()
    st.markdown(Transcript)

