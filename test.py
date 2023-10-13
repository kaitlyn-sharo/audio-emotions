from audio_emotions.audio_emotions.emotion_recognition import EmotionRecognizer
from sklearn.svm import SVC
emotion_model = SVC(probability=True)
recognizer = EmotionRecognizer(model=emotion_model, emotions=['sad', 'angry', 'calm', 'disgust', 'ps', 'happy', "fear"], balance=True, verbose=0)
recognizer.train()