from gtts import gTTS
import os
text = "Natural language processing is good for design and developemnt"

language = 'es'

speech = gTTS(text = text, lang = language)

speech.save("text.mp3")
os.system("start text.mp3")
