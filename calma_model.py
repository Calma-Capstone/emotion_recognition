
import os
import tensorflow as tf
import numpy as np
import pathlib
import librosa

class CalmaModel():
    
    def __init__(self, model_path=pathlib.Path("models/"), isCloud=False):
        self.model_path = model_path
        self.isCloud = isCloud
        self.model = None
        self.model_version = None
        
    def load_model(self, version="v2.2"):
        if version == "latest": 
            version = os.listdir(self.model_path)[-1]
        if self.model_version == version:
            self.model = tf.keras.models.load_model(os.path.join(self.model_path, version))
        return self.model
        
    def __extract_features(self, data, sample_rate):
        # ZCR
        result = np.array([])
        zcr = librosa.feature.zero_crossing_rate(y=data)
        zcr_mean = np.mean(zcr, axis=1)
        zcr_min = np.min(zcr, axis=1)
        zcr_max = np.max(zcr, axis=1)
        zcr_feature = np.concatenate((zcr_min, zcr_mean, zcr_max))

        # Chroma_stft
        # stft = np.abs(librosa.stft(data))
        
        chroma_stft = librosa.feature.chroma_stft(S=librosa.stft(data), sr=sample_rate)
        chroma_stft_mean = np.mean(chroma_stft, axis=1)
        chroma_stft_min = np.min(chroma_stft, axis=1)
        chroma_stft_max = np.max(chroma_stft, axis=1)
        chroma_stft_feature = np.concatenate((chroma_stft_min, chroma_stft_mean, chroma_stft_max))

        # MFCC
        mfcc = librosa.feature.mfcc(y=data, sr=sample_rate)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_min = np.min(mfcc, axis=1)
        mfcc_max = np.max(mfcc, axis=1)
        mfcc_feature = np.concatenate((mfcc_min, mfcc_mean, mfcc_max))

        # Root Mean Square Value
        # rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
        # result = np.hstack((result, rms)) # stacking horizontally

        # MelSpectogram
        mel = librosa.feature.melspectrogram(y=data, sr=sample_rate)
        mel_mean = np.mean(mel, axis=1)
        mel_min = np.min(mel, axis=1)
        mel_max = np.max(mel, axis=1)
        mel_feature = np.concatenate((mel_min, mel_mean, mel_max))
        
        tonnetz = librosa.feature.tonnetz(y=data, sr=sample_rate)
        tonnetz_mean = np.mean(tonnetz, axis=1)
        tonnetz_min = np.min(tonnetz, axis=1)
        tonnetz_max = np.max(tonnetz, axis=1)
        tonnetz_feature = np.concatenate((tonnetz_min, tonnetz_mean, tonnetz_max))
        
        result = np.concatenate((zcr_feature, chroma_stft_feature, mfcc_feature, mel_feature, tonnetz_feature))
        return result
    
    def __process_audio_to_predict(self, path):
        data, sr = librosa.load(path, duration=2.5)
        return self.__extract_features(data, sr).real
    
    def __predict_audio(self, audio_features):
        emotions = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        if self.model == None: print("model was not loaded"); return
        return emotions[np.argmax(self.model.predict(np.expand_dims(audio_features,0)), axis=1)[0]]
    
    def predict(self, audio_path):
        features =  self.__process_audio_features(audio_path)
        return self.__predict_audio(features)
    
    def predict_long_audio(self, audio_path):
        # TODO: find best solution for long audio to predict the emotion
        pass
    
    def add_new_model(self, file, version):
        # TODO: add new model to our repo
        pass