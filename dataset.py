from gtts import gTTS
import os

text = "To be or not to be, that is the question!"
tts = gTTS(text=text, lang='en')

filename = "output.mp3"
tts.save(filename)

# Play the mp3 file (Linux example)
os.system(f"mpg123 {filename}")
