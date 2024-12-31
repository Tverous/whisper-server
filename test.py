import whisper

model = whisper.load_model("base")
result = model.transcribe("prompt.wav")
print(result["text"])