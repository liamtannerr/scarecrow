from piper.voice import PiperVoice, SynthesisConfig
import sounddevice as sd

# Configure Voice
MODEL_PATH = "./models/piper/en_GB-alan-medium.onnx"
TEXT = "Hello I am a pumpkin. oooooooo spooky right?"

voice = PiperVoice.load(MODEL_PATH)

# Configure audio output
sr = voice.config.sample_rate  # Piper tells us the model's sample rate
cfg = SynthesisConfig(length_scale=1.2, noise_scale=2)
sd.default.samplerate = sr

sd.default.channels = 1

# Stream synthesis and play live
with sd.OutputStream(dtype="int16", channels=1, samplerate=sr) as stream:
    for chunk in voice.synthesize(  text=TEXT, syn_config=cfg):
        stream.write(chunk.audio_int16_array)


