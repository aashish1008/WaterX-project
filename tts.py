from gtts import gTTS
import os
import time
import playsound


class TTS:
    def __init__(self, ans, lang='en'):
        self.ans = ans
        self.lang = lang
        self.lang_opt = {
            'af': 'Afrikaans',
            'am': 'Amharic',
            'ar': 'Arabic',
            'bg': 'Bulgarian',
            'bn': 'Bengali',
            'bs': 'Bosnian',
            'ca': 'Catalan',
            'cs': 'Czech',
            'cy': 'Welsh',
            'da': 'Danish',
            'de': 'German',
            'el': 'Greek',
            'en': 'English',
            'es': 'Spanish',
            'et': 'Estonian',
            'eu': 'Basque',
            'fi': 'Finnish',
            'fr': 'French',
            'gl': 'Galician',
            'gu': 'Gujarati',
            'ha': 'Hausa',
            'hi': 'Hindi',
            'hr': 'Croatian',
            'hu': 'Hungarian',
            'id': 'Indonesian',
            'is': 'Icelandic',
            'it': 'Italian',
            'iw': 'Hebrew',
            'ja': 'Japanese',
            'jw': 'Javanese',
            'km': 'Khmer',
            'kn': 'Kannada',
            'ko': 'Korean',
            'la': 'Latin',
            'lt': 'Lithuanian',
            'lv': 'Latvian',
            'ml': 'Malayalam',
            'mr': 'Marathi',
            'ms': 'Malay',
            'my': 'Myanmar (Burmese)',
            'ne': 'Nepali',
            'nl': 'Dutch',
            'no': 'Norwegian',
            'pa': 'Punjabi (Gurmukhi)',
            'pl': 'Polish',
            'pt': 'Portuguese (Brazil)',
            'pt-PT': 'Portuguese (Portugal)',
            'ro': 'Romanian',
            'ru': 'Russian',
            'si': 'Sinhala',
            'sk': 'Slovak',
            'sq': 'Albanian',
            'sr': 'Serbian',
            'su': 'Sundanese',
            'sv': 'Swedish',
            'sw': 'Swahili',
            'ta': 'Tamil',
            'te': 'Telugu',
            'th': 'Thai',
            'tl': 'Filipino',
            'tr': 'Turkish',
            'uk': 'Ukrainian',
            'ur': 'Urdu',
            'vi': 'Vietnamese',
            'yue': 'Cantonese',
            'zh-CN': 'Chinese (Simplified)',
            'zh-TW': 'Chinese (Mandarin / Taiwan)',
            'zh': 'Chinese (Mandarin)'
        }

    def txt_to_speech(self):
        if self.lang in self.lang_opt:
            tts = gTTS(text=self.ans, lang=self.lang, slow=False)

            # Save the audio file
            filename = 'speech.mp3'
            tts.save(filename)

            # Play the audio file
            print("Playing speech...")
            playsound.playsound(filename)

            # Ensure the file is removed after a short delay to avoid issues with file locking
            time.sleep(1.5)  # Adjust timing to ensure the sound finishes playing
            if os.path.exists(filename):
                os.remove(filename)
        else:
            print(f"Language '{self.lang}' is not supported.")


if __name__ == "__main__":
    ans = "உன் பெயர் என்ன?"  # Example in Tamil
    play = TTS(ans, lang="ta")
    play.txt_to_speech()
