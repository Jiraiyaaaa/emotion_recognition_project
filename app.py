import cv2
import time
import mediapipe as mp
from collections import Counter, deque
import numpy as np
import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import threading
import queue
import json
import sounddevice as sd
from functools import lru_cache
import torch
import torch.nn as nn
import asyncio
import logging
from datetime import datetime
import psutil
import argparse
import sys
import base64
from typing import List

# Web framework imports (only imported when needed)
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False
    print("FastAPI not available. Web interface disabled.")

# Audio processing imports
try:
    import opensmile
    from vosk import Model, KaldiRecognizer
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Audio processing libraries not available.")

# DeepFace import
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("DeepFace not available.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Enable OpenCV DNN acceleration if available
try:
    cv2.dnn.DNN_TARGET_CUDA
    cv2.dnn.DNN_BACKEND_CUDA
    print("CUDA acceleration enabled for OpenCV DNN")
except:
    print("CUDA not available, using CPU")

# Configure Gemini API
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    gemini_model_text = genai.GenerativeModel('gemini-1.5-flash')
    gemini_model_vision = genai.GenerativeModel('gemini-pro-vision')
    GEMINI_AVAILABLE = True
    print("Gemini API configured successfully.")
except (KeyError, TypeError):
    GEMINI_AVAILABLE = False
    print("Warning: GEMINI_API_KEY not found. AI coach features will be disabled.")

# Configure speech recognition
if AUDIO_AVAILABLE:
    try:
        vosk_model = Model("vosk-model-small-en-in-0.4")
        recognizer = KaldiRecognizer(vosk_model, 16000)
        sentiment_analyzer = SentimentIntensityAnalyzer()
        SPEECH_AVAILABLE = True
        print("Speech recognition models loaded successfully.")
    except Exception as e:
        SPEECH_AVAILABLE = False
        print(f"Warning: Could not load speech models. Error: {e}")
else:
    SPEECH_AVAILABLE = False

# ================================================================================================
# CORE CLASSES - All autism-specific emotion recognition components
# ================================================================================================

class PerformanceMonitor:
    """Real-time performance monitoring and optimization"""
    def __init__(self):
        self.fps_history = deque(maxlen=30)
        self.latency_history = deque(maxlen=30)
        self.memory_usage = deque(maxlen=30)
        self.last_frame_time = time.time()
        
    def update(self, processing_time):
        current_time = time.time()
        if hasattr(self, 'last_frame_time'):
            fps = 1.0 / (current_time - self.last_frame_time)
            self.fps_history.append(fps)
        
        self.latency_history.append(processing_time * 1000)  # Convert to ms
        self.memory_usage.append(psutil.virtual_memory().percent)
        self.last_frame_time = current_time
        
    def get_metrics(self):
        return {
            'fps': np.mean(self.fps_history) if self.fps_history else 0,
            'latency_ms': np.mean(self.latency_history) if self.latency_history else 0,
            'memory_percent': np.mean(self.memory_usage) if self.memory_usage else 0,
            'status': self.get_system_status()
        }
    
    def get_system_status(self):
        if not self.latency_history:
            return "STARTING"
        avg_latency = np.mean(self.latency_history)
        if avg_latency < 50:
            return "OPTIMAL"
        elif avg_latency < 100:
            return "GOOD"
        elif avg_latency < 200:
            return "MODERATE"
        else:
            return "SLOW"


class GeminiLiveProcessor:
    """Enhanced Gemini API integration with caching and rate limiting"""
    def __init__(self):
        self.cache = {}
        self.last_request_time = {}
        self.min_request_interval = 2.0  # seconds
        self.session_context = []
        
    @lru_cache(maxsize=100)
    def get_cached_response(self, prompt_key):
        try:
            if GEMINI_AVAILABLE:
                response = gemini_model_text.generate_content(prompt_key)
                return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API Error: {e}")
            return None
        
    def check_sarcasm(self, text, emotion, sentiment):
        """Detect potential sarcasm from emotion/sentiment mismatch"""
        if not GEMINI_AVAILABLE or not text:
            return None
            
        is_mismatch = (sentiment == "Positive" and emotion in ["Frustrated", "Overwhelmed"]) or \
                     (sentiment == "Negative" and emotion in ["Happy", "Excited"])
        
        if is_mismatch:
            current_time = time.time()
            cache_key = f"sarcasm_{text}_{emotion}_{sentiment}"
            
            if (cache_key not in self.last_request_time or 
                current_time - self.last_request_time[cache_key] > self.min_request_interval):
                
                prompt = f"Text: '{text}' | Emotion: {emotion} | Sentiment: {sentiment}. Sarcasm likely? Brief response."
                response = self.get_cached_response(prompt)
                self.last_request_time[cache_key] = current_time
                return response
        return None
    
    def get_emotion_interpretation(self, emotion, monotony_score, context=""):
        """Get contextual emotion interpretation"""
        if not GEMINI_AVAILABLE:
            return ""
            
        current_time = time.time()
        cache_key = f"interpret_{emotion}_{monotony_score:.1f}_{context}"
        
        if (cache_key not in self.last_request_time or 
            current_time - self.last_request_time[cache_key] > self.min_request_interval * 2):
            
            prompt = f"Person shows {emotion} emotion, monotony score {monotony_score:.1f}. {context}. Brief insight:"
            response = self.get_cached_response(prompt)
            self.last_request_time[cache_key] = current_time
            return response or ""
        return ""
    
    def get_suggestions(self, emotion, context=""):
        """Get actionable suggestions"""
        if not GEMINI_AVAILABLE:
            return ""
            
        current_time = time.time()
        cache_key = f"suggest_{emotion}_{context}"
        
        if (cache_key not in self.last_request_time or 
            current_time - self.last_request_time[cache_key] > self.min_request_interval):
            
            prompt = f"Person is {emotion}. {context}. Give 2 brief interaction tips:"
            response = self.get_cached_response(prompt)
            self.last_request_time[cache_key] = current_time
            return response or ""
        return ""


class OptimizedCALMEDAutismCNN:
    """Enhanced autism-specific CNN with optimizations"""
    def __init__(self):
        self.autism_emotions = ['calm', 'happy', 'frustrated', 'overwhelmed', 'focused', 'anxious', 'excited', 'neutral']
        self.emotion_map = {
            'neutral': 'calm', 'happy': 'happy', 'angry': 'frustrated', 'disgust': 'frustrated',
            'sad': 'overwhelmed', 'fear': 'anxious', 'surprise': 'excited'
        }
        self.target_size = (224, 224)
        
    def preprocess_frame(self, frame):
        """Optimized preprocessing with autism-specific attention weighting"""
        resized = cv2.resize(frame, self.target_size)
        
        # Apply autism attention weighting
        h, w, _ = resized.shape
        mask = np.ones_like(resized, dtype=np.float32)
        mask[0:int(h*0.4), :] *= 0.8  # Reduce eye region weight
        mask[int(h*0.6):, :] *= 1.3   # Increase mouth region weight
        
        return (resized * mask).astype(np.uint8)
    
    def analyze(self, frame):
        """Optimized emotion analysis"""
        if not DEEPFACE_AVAILABLE:
            return {
                'emotion': "DeepFace unavailable",
                'confidence': 0.0,
                'scores': {},
                'processing_time': 0.001
            }
            
        try:
            start_time = time.time()
            processed_frame = self.preprocess_frame(frame)
            
            analysis = DeepFace.analyze(
                processed_frame, 
                actions=['emotion'], 
                detector_backend='opencv',
                enforce_detection=False
            )
            
            processing_time = time.time() - start_time
            
            if analysis:
                dominant_emotion = analysis[0]['dominant_emotion']
                autism_emotion = self.emotion_map.get(dominant_emotion, 'focused').capitalize()
                confidence = analysis[0]['emotion'][dominant_emotion] / 100.0
                
                return {
                    'emotion': autism_emotion,
                    'confidence': confidence,
                    'scores': analysis[0]['emotion'],
                    'processing_time': processing_time
                }
        except Exception as e:
            logger.warning(f"Emotion analysis failed: {e}")
            
        return {
            'emotion': "Unknown",
            'confidence': 0.0,
            'scores': {},
            'processing_time': 0.001
        }


class EnhancedAutismAudioProcessor:
    """Improved audio processing with overlapping windows"""
    def __init__(self, state):
        self.state = state
        self.audio_queue = queue.Queue()
        self.audio_buffer = deque(maxlen=48000)  # 3 seconds at 16kHz
        self.last_prosody_time = time.time()
        self.prosody_interval = 0.5  # Process every 500ms
        
        if AUDIO_AVAILABLE:
            self.smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPS, 
                feature_level=opensmile.FeatureLevel.Functionals
            )
        
    def extract_enhanced_prosody(self, audio_chunk):
        """Enhanced prosody analysis with multiple features"""
        if not AUDIO_AVAILABLE:
            return
            
        try:
            if len(audio_chunk) < 8000:  # Minimum chunk size
                return
                
            features = self.smile.process_signal(audio_chunk, 16000)
            
            f0_std = features['F0semitoneFrom27.5Hz_sma3nz_stddevNorm'].values[0]
            loudness_std = features['loudness_sma3_stddevNorm'].values[0]
            jitter = features['jitterLocal_sma3nz_amean'].values[0]
            
            # Calculate autism-specific metrics
            monotony_score = max(0.0, min(1.0, 1.0 - (f0_std + loudness_std) / 2.0))
            voice_stress = min(jitter * 10, 1.0)
            
            # Update state with multiple metrics
            self.state.update({
                'monotony_score': monotony_score,
                'voice_stress': voice_stress,
                'f0_variability': f0_std,
                'loudness_variability': loudness_std,
                'last_audio_update': time.time()
            })
            
        except Exception as e:
            logger.warning(f"Prosody extraction failed: {e}")
            self.state.update({'monotony_score': 0.0, 'voice_stress': 0.0})
    
    def process_audio_stream(self):
        """Enhanced audio stream processing"""
        if not AUDIO_AVAILABLE:
            return
            
        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio stream warning: {status}")
            self.audio_queue.put(indata.copy())
            
        try:
            with sd.InputStream(samplerate=16000, channels=1, dtype='float32', callback=audio_callback):
                while True:
                    audio_chunk = self.audio_queue.get()
                    
                    # Add to buffer for overlapping window analysis
                    self.audio_buffer.extend(audio_chunk.flatten())
                    
                    # Process speech recognition
                    if SPEECH_AVAILABLE:
                        audio_int16 = (audio_chunk * 32767).astype(np.int16)
                        if recognizer.AcceptWaveform(audio_int16.tobytes()):
                            result = json.loads(recognizer.Result())
                            if result.get('text'):
                                self.state['transcribed_text'] = result['text']
                                score = sentiment_analyzer.polarity_scores(self.state['transcribed_text'])
                                if score['compound'] >= 0.05:
                                    self.state['text_sentiment'] = "Positive"
                                elif score['compound'] <= -0.05:
                                    self.state['text_sentiment'] = "Negative"
                                else:
                                    self.state['text_sentiment'] = "Neutral"
                    
                    # Process prosody with timing control
                    current_time = time.time()
                    if (current_time - self.last_prosody_time > self.prosody_interval and 
                        len(self.audio_buffer) >= 8000):
                        
                        prosody_chunk = np.array(list(self.audio_buffer))
                        if np.linalg.norm(prosody_chunk) > 0.01:
                            self.extract_enhanced_prosody(prosody_chunk)
                        else:
                            self.state.update({'monotony_score': 0.0, 'voice_stress': 0.0})
                        
                        self.last_prosody_time = current_time
                        
        except Exception as e:
            logger.error(f"Audio processing error: {e}")


class AdvancedPersonalizationEngine:
    """Enhanced personalization with adaptive learning"""
    def __init__(self, user_profile=None):
        profile = user_profile or {}
        self.severity = profile.get('severity', 'moderate')
        
        # Adaptive parameters
        self.confidence_thresholds = {'mild': 0.75, 'moderate': 0.65, 'severe': 0.55}
        self.temporal_windows_s = {'mild': 0.15, 'moderate': 0.2, 'severe': 0.25}
        
        self.current_threshold = self.confidence_thresholds[self.severity]
        self.temporal_window = self.temporal_windows_s[self.severity]
        
        # Enhanced tracking
        self.emotion_history = deque(maxlen=20)
        self.confidence_history = deque(maxlen=10)
        self.user_feedback_history = []
        
        # Learning parameters
        self.adaptation_rate = 0.1
        self.stability_factor = 0.8
        
    def adapt_predictions(self, emotion_result, timestamp):
        """Advanced prediction adaptation with confidence weighting"""
        if not emotion_result or not emotion_result.get('scores'):
            return "N/A"
            
        emotion = emotion_result['emotion']
        confidence = emotion_result['confidence']
        scores = emotion_result['scores']
        
        # Add to history with metadata
        self.emotion_history.append({
            'emotion': emotion,
            'confidence': confidence,
            'timestamp': timestamp,
            'scores': scores
        })
        
        self.confidence_history.append(confidence)
        
        # Clean old entries
        cutoff = timestamp - self.temporal_window
        self.emotion_history = deque([
            h for h in self.emotion_history if h['timestamp'] > cutoff
        ], maxlen=20)
        
        if len(self.emotion_history) < 2:
            return emotion
            
        # Advanced weighted smoothing
        weighted_emotions = Counter()
        total_weight = 0
        
        for item in self.emotion_history:
            age_weight = 1.0 - (timestamp - item['timestamp']) / self.temporal_window
            confidence_weight = item['confidence']
            stability_bonus = self.stability_factor if item['emotion'] == emotion else 1.0
            
            final_weight = age_weight * confidence_weight * stability_bonus
            weighted_emotions[item['emotion']] += final_weight
            total_weight += final_weight
        
        # Get most stable emotion
        if total_weight > 0:
            final_emotion = weighted_emotions.most_common(1)[0][0]
            
            # Adaptive threshold adjustment
            avg_confidence = np.mean(self.confidence_history)
            if avg_confidence > 0.8:
                self.current_threshold = min(0.9, self.current_threshold + 0.05)
            elif avg_confidence < 0.4:
                self.current_threshold = max(0.3, self.current_threshold - 0.05)
            
            return final_emotion
            
        return emotion


class SessionAnalyzer:
    """Session analytics and logging"""
    def __init__(self):
        self.session_start = time.time()
        self.emotion_log = []
        self.interaction_log = []
        self.performance_log = []
        
    def log_emotion(self, emotion_data):
        """Log emotion detection event"""
        self.emotion_log.append({
            'timestamp': time.time(),
            'emotion': emotion_data.get('emotion'),
            'confidence': emotion_data.get('confidence'),
            'processing_time': emotion_data.get('processing_time')
        })
    
    def log_interaction(self, interaction_type, details):
        """Log user interaction"""
        self.interaction_log.append({
            'timestamp': time.time(),
            'type': interaction_type,
            'details': details
        })
    
    def log_performance(self, metrics):
        """Log performance metrics"""
        self.performance_log.append({
            'timestamp': time.time(),
            'metrics': metrics.copy()
        })
    
    def generate_summary(self):
        """Generate session summary"""
        session_duration = time.time() - self.session_start
        
        if not self.emotion_log:
            return "No emotion data recorded during session."
        
        emotions = [log['emotion'] for log in self.emotion_log if log['emotion']]
        emotion_counts = Counter(emotions)
        
        avg_confidence = np.mean([log['confidence'] for log in self.emotion_log if log['confidence']])
        avg_processing_time = np.mean([log['processing_time'] for log in self.emotion_log if log['processing_time']])
        
        summary = f"""
Session Summary ({session_duration/60:.1f} minutes):
- Total emotions detected: {len(self.emotion_log)}
- Most common emotions: {', '.join([f"{emo}({count})" for emo, count in emotion_counts.most_common(3)])}
- Average confidence: {avg_confidence:.1%}
- Average processing time: {avg_processing_time*1000:.1f}ms
- Interactions logged: {len(self.interaction_log)}
        """
        return summary.strip()


# ================================================================================================
# DESKTOP APPLICATION - OpenCV-based interface
# ================================================================================================

class EnhancedUIManager:
    """Improved UI with performance metrics and better layout"""
    def __init__(self, panel_width=500):
        self.panel_width = panel_width
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.colors = {
            "bg": (25, 25, 25), "text": (255, 255, 255), "header": (120, 120, 120),
            "highlight": (0, 255, 255), "ok": (0, 255, 0), "warn": (0, 165, 255),
            "error": (0, 0, 255), "accent": (255, 165, 0)
        }
        
    def create_combined_frame(self, video_frame):
        """Optimized frame creation with pre-rendered panel"""
        h, w, _ = video_frame.shape
        combined = np.zeros((h, w + self.panel_width, 3), dtype=np.uint8)
        combined[:, :w] = video_frame
        combined[:, w:] = self.colors["bg"]
        return combined
    
    def draw_performance_metrics(self, frame, metrics):
        """Draw real-time performance metrics"""
        w = frame.shape[1]
        base_x = w - self.panel_width + 20
        
        # Performance header
        cv2.putText(frame, "System Performance", (base_x, 30), self.font, 0.8, self.colors["header"], 2)
        
        # FPS
        fps_color = self.colors["ok"] if metrics['fps'] > 20 else self.colors["warn"] if metrics['fps'] > 10 else self.colors["error"]
        cv2.putText(frame, f"FPS: {metrics['fps']:.1f}", (base_x, 55), self.font, 0.6, fps_color, 1)
        
        # Latency
        latency_color = self.colors["ok"] if metrics['latency_ms'] < 50 else self.colors["warn"] if metrics['latency_ms'] < 100 else self.colors["error"]
        cv2.putText(frame, f"Latency: {metrics['latency_ms']:.0f}ms", (base_x + 120, 55), self.font, 0.6, latency_color, 1)
        
        # Memory
        memory_color = self.colors["ok"] if metrics['memory_percent'] < 70 else self.colors["warn"] if metrics['memory_percent'] < 85 else self.colors["error"]
        cv2.putText(frame, f"Memory: {metrics['memory_percent']:.0f}%", (base_x + 250, 55), self.font, 0.6, memory_color, 1)
        
        # Status
        status_colors = {"OPTIMAL": self.colors["ok"], "GOOD": self.colors["highlight"], "MODERATE": self.colors["warn"], "SLOW": self.colors["error"]}
        status_color = status_colors.get(metrics['status'], self.colors["text"])
        cv2.putText(frame, f"Status: {metrics['status']}", (base_x + 350, 55), self.font, 0.6, status_color, 1)
        
        # Separator line
        cv2.line(frame, (base_x, 70), (w - 20, 70), self.colors["header"], 1)
    
    def draw_enhanced_panel_header(self, frame, text):
        """Enhanced panel header with timestamp"""
        w = frame.shape[1]
        base_x = w - self.panel_width + 20
        
        cv2.putText(frame, text, (base_x, 100), self.font, 1.0, self.colors["text"], 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (w - 120, 100), self.font, 0.6, self.colors["header"], 1)
        
        cv2.line(frame, (base_x, 115), (w - 20, 115), self.colors["header"], 2)
    
    def draw_enhanced_cue(self, frame, label, value, y_pos, ok_value=None, format_func=None):
        """Enhanced cue display with formatting"""
        w = frame.shape[1]
        base_x = w - self.panel_width + 20
        
        cv2.putText(frame, label, (base_x, y_pos), self.font, 0.8, self.colors["text"], 2)
        
        # Format value if function provided
        display_value = format_func(value) if format_func else str(value)
        
        # Color coding
        color = self.colors["text"]
        if ok_value is not None:
            color = self.colors["ok"] if value == ok_value else self.colors["error"]
        elif isinstance(value, (int, float)):
            if value > 0.7:
                color = self.colors["error"]
            elif value > 0.4:
                color = self.colors["warn"]
            else:
                color = self.colors["ok"]
        
        cv2.putText(frame, display_value, (base_x + 200, y_pos), self.font, 0.8, color, 2)
    
    def draw_multimodal_analysis(self, frame, emotion_data, audio_data, y_pos):
        """Draw comprehensive multimodal analysis"""
        w = frame.shape[1]
        base_x = w - self.panel_width + 20
        
        cv2.putText(frame, "Multimodal Analysis", (base_x, y_pos), self.font, 0.8, self.colors["text"], 2)
        
        current_y = y_pos + 25
        
        # Emotion with confidence
        if emotion_data:
            emotion = emotion_data.get('emotion', 'Unknown')
            confidence = emotion_data.get('confidence', 0.0)
            cv2.putText(frame, f"Emotion: {emotion} ({confidence:.0%})", 
                       (base_x, current_y), self.font, 0.7, self.colors["highlight"], 1)
            current_y += 25
        
        # Audio metrics
        if audio_data:
            monotony = audio_data.get('monotony_score', 0.0)
            stress = audio_data.get('voice_stress', 0.0)
            
            monotony_color = self.colors["error"] if monotony > 0.7 else self.colors["warn"] if monotony > 0.4 else self.colors["ok"]
            cv2.putText(frame, f"Monotony: {monotony:.2f}", 
                       (base_x, current_y), self.font, 0.7, monotony_color, 1)
            
            stress_color = self.colors["error"] if stress > 0.7 else self.colors["warn"] if stress > 0.4 else self.colors["ok"]
            cv2.putText(frame, f"Voice Stress: {stress:.2f}", 
                       (base_x + 150, current_y), self.font, 0.7, stress_color, 1)
            current_y += 25
    
    def draw_ai_coach_section(self, frame, interpretation, suggestions, sarcasm_alert):
        """Enhanced AI coach section with multiple types of feedback"""
        h, w, _ = frame.shape
        base_x = w - self.panel_width + 20
        base_y = h - 280
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (base_x - 10, base_y - 20), (w - 10, h - 10), (40, 40, 40), -1)
        frame[:] = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        current_y = base_y
        
        # Sarcasm alert (highest priority)
        if sarcasm_alert:
            cv2.putText(frame, "Sarcasm Alert", (base_x, current_y), self.font, 0.8, self.colors["error"], 2)
            current_y += 25
            words = sarcasm_alert.split(' ')
            self._draw_wrapped_text(frame, words, base_x, current_y, 35, 4)
            current_y += 120
        
        # Interpretation
        elif interpretation:
            cv2.putText(frame, "AI Interpretation", (base_x, current_y), self.font, 0.8, self.colors["highlight"], 2)
            current_y += 25
            words = interpretation.split(' ')
            self._draw_wrapped_text(frame, words, base_x, current_y, 35, 4)
            current_y += 120
        
        # Suggestions
        elif suggestions:
            cv2.putText(frame, "Suggestions", (base_x, current_y), self.font, 0.8, self.colors["accent"], 2)
            current_y += 25
            words = suggestions.split(' ')
            self._draw_wrapped_text(frame, words, base_x, current_y, 35, 4)
    
    def _draw_wrapped_text(self, frame, words, x, y, max_chars_per_line, max_lines):
        """Helper to draw wrapped text"""
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= max_chars_per_line:
                current_line += " " + word if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        for i, line in enumerate(lines[:max_lines]):
            cv2.putText(frame, line, (x, y + i * 25), self.font, 0.7, self.colors["text"], 1)


class OptimizedAutismEmotionSystem:
    """Complete optimized autism emotion recognition system - Desktop Version"""
    def __init__(self):
        # Core components
        self.emotion_model = OptimizedCALMEDAutismCNN()
        self.personalization = AdvancedPersonalizationEngine()
        self.gemini_processor = GeminiLiveProcessor()
        self.performance_monitor = PerformanceMonitor()
        self.session_analyzer = SessionAnalyzer()
        
        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,  # Focus on single face for performance
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # State management
        self.state = {
            'analysis_cache': [],
            'monotony_score': 0.0,
            'voice_stress': 0.0,
            'transcribed_text': '',
            'text_sentiment': 'Neutral',
            'last_audio_update': 0,
            'current_emotion': 'Unknown',
            'emotion_confidence': 0.0,
            'system_status': 'Starting'
        }
        
        # Threading
        self.vision_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.running = True
        
        # UI state
        self.last_interpretation_time = 0
        self.last_suggestion_time = 0
        self.current_interpretation = ""
        self.current_suggestions = ""
        self.current_sarcasm_alert = ""
        self.sarcasm_alert_end_time = 0
        
        # Performance optimization
        self.analysis_interval = 3  # Process every 3rd frame for performance
        self.frame_count = 0
        
    def vision_worker(self):
        """Optimized vision processing worker"""
        logger.info("Vision worker started")
        
        while self.running:
            try:
                frame = self.vision_queue.get(timeout=1.0)
                if frame is None:
                    break
                
                start_time = time.time()
                cache = []
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    # Process only the first face for performance
                    face_landmarks = results.multi_face_landmarks[0]
                    h, w, _ = frame.shape
                    
                    # Get face bounding box
                    x_coords = [lm.x * w for lm in face_landmarks.landmark]
                    y_coords = [lm.y * h for lm in face_landmarks.landmark]
                    
                    x_min, x_max = int(min(x_coords) - 20), int(max(x_coords) + 20)
                    y_min, y_max = int(min(y_coords) - 20), int(max(y_coords) + 20)
                    
                    # Ensure bounds are within frame
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(w, x_max), min(h, y_max)
                    
                    face_img = frame[y_min:y_max, x_min:x_max]
                    
                    if face_img.size > 0:
                        # Analyze emotion
                        emotion_result = self.emotion_model.analyze(face_img)
                        
                        # Apply personalization
                        final_emotion = self.personalization.adapt_predictions(
                            emotion_result, time.time()
                        )
                        
                        # Update emotion result
                        emotion_result['emotion'] = final_emotion
                        
                        # Log for analytics
                        self.session_analyzer.log_emotion(emotion_result)
                        
                        cache.append({
                            'box': (x_min, y_min, x_max - x_min, y_max - y_min),
                            'emotion_data': emotion_result
                        })
                
                processing_time = time.time() - start_time
                self.result_queue.put((cache, processing_time))
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Vision worker error: {e}")
                self.result_queue.put(([], 0.001))
        
        logger.info("Vision worker stopped")
    
    def update_ai_coaching(self):
        """Update AI coaching with rate limiting"""
        current_time = time.time()
        
        # Check for sarcasm (high priority)
        if (self.state.get('transcribed_text') and 
            current_time > self.sarcasm_alert_end_time):
            
            sarcasm_result = self.gemini_processor.check_sarcasm(
                self.state['transcribed_text'],
                self.state['current_emotion'],
                self.state['text_sentiment']
            )
            
            if sarcasm_result:
                self.current_sarcasm_alert = sarcasm_result
                self.sarcasm_alert_end_time = current_time + 5.0
                self.state['transcribed_text'] = ""  # Clear to prevent repeated checks
        
        # Get interpretation (medium priority)
        elif (current_time - self.last_interpretation_time > 8.0 and 
              self.state['current_emotion'] != 'Unknown'):
            
            context = f"monotony {self.state['monotony_score']:.1f}"
            interpretation = self.gemini_processor.get_emotion_interpretation(
                self.state['current_emotion'],
                self.state['monotony_score'],
                context
            )
            
            if interpretation:
                self.current_interpretation = interpretation
                self.last_interpretation_time = current_time
    
    def handle_key_input(self, key):
        """Handle keyboard input"""
        if key == ord('h'):  # Help suggestions
            if self.state['current_emotion'] != 'Unknown':
                suggestions = self.gemini_processor.get_suggestions(
                    self.state['current_emotion'],
                    f"voice stress {self.state['voice_stress']:.1f}"
                )
                if suggestions:
                    self.current_suggestions = suggestions
                    self.last_suggestion_time = time.time()
                    
                self.session_analyzer.log_interaction('help_request', {
                    'emotion': self.state['current_emotion']
                })
        
        elif key == ord('c'):  # Clear AI coaching
            self.current_interpretation = ""
            self.current_suggestions = ""
            self.current_sarcasm_alert = ""
            self.sarcasm_alert_end_time = 0
            
        elif key == ord('s'):  # Save session summary
            summary = self.session_analyzer.generate_summary()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"autism_session_{timestamp}.txt"
            
            try:
                with open(filename, 'w') as f:
                    f.write(summary)
                print(f"Session summary saved to {filename}")
            except Exception as e:
                logger.error(f"Failed to save summary: {e}")
    
    def main_loop(self):
        """Main application loop with all optimizations"""
        logger.info("Starting Optimized Autism Emotion Recognition System - Desktop Mode")
        
        # Start worker threads
        vision_thread = threading.Thread(target=self.vision_worker, daemon=True)
        vision_thread.start()
        
        if AUDIO_AVAILABLE:
            audio_processor = EnhancedAutismAudioProcessor(self.state)
            audio_thread = threading.Thread(target=audio_processor.process_audio_stream, daemon=True)
            audio_thread.start()
        
        # Initialize camera with optimizations
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        
        if not cap.isOpened():
            logger.error("Failed to open camera")
            return
        
        # UI setup
        window_name = 'Optimized Autism Support System'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        ui = EnhancedUIManager()
        
        print("\n=== DESKTOP CONTROLS ===")
        print("'q' - Quit application")
        print("'h' - Get help suggestions")
        print("'c' - Clear AI coaching display")
        print("'s' - Save session summary")
        print("========================\n")
        
        try:
            while self.running:
                loop_start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    continue
                
                frame = cv2.flip(frame, 1)
                self.frame_count += 1
                
                # Process vision every N frames for performance
                if (self.frame_count % self.analysis_interval == 0 and 
                    self.vision_queue.qsize() < 2):
                    self.vision_queue.put(frame.copy())
                
                # Get vision results
                processing_time = 0.001
                try:
                    cache, processing_time = self.result_queue.get_nowait()
                    self.state['analysis_cache'] = cache
                    
                    # Update current emotion state
                    if cache:
                        emotion_data = cache[0]['emotion_data']
                        self.state['current_emotion'] = emotion_data['emotion']
                        self.state['emotion_confidence'] = emotion_data['confidence']
                        
                except queue.Empty:
                    pass
                
                # Update performance metrics
                self.performance_monitor.update(processing_time)
                current_metrics = self.performance_monitor.get_metrics()
                self.session_analyzer.log_performance(current_metrics)
                
                # Update AI coaching
                self.update_ai_coaching()
                
                # Create display frame
                display_frame = ui.create_combined_frame(frame)
                
                # Draw UI components
                ui.draw_performance_metrics(display_frame, current_metrics)
                ui.draw_enhanced_panel_header(display_frame, "Autism Support Analysis")
                
                # Draw emotion analysis
                emotion_data = None
                if self.state['analysis_cache']:
                    emotion_data = self.state['analysis_cache'][0]['emotion_data']
                
                audio_data = {
                    'monotony_score': self.state.get('monotony_score', 0.0),
                    'voice_stress': self.state.get('voice_stress', 0.0)
                }
                
                ui.draw_multimodal_analysis(display_frame, emotion_data, audio_data, 140)
                
                # Draw face detection boxes and labels
                for result in self.state['analysis_cache']:
                    x, y, w, h = result['box']
                    emotion_data = result['emotion_data']
                    
                    # Draw bounding box
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Draw emotion label with confidence
                    label = f"{emotion_data['emotion']} ({emotion_data['confidence']:.0%})"
                    cv2.putText(display_frame, label, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Draw AI coaching section
                current_time = time.time()
                show_sarcasm = current_time < self.sarcasm_alert_end_time
                show_suggestions = (current_time - self.last_suggestion_time < 10.0 and 
                                   self.current_suggestions)
                
                sarcasm_alert = self.current_sarcasm_alert if show_sarcasm else ""
                suggestions = self.current_suggestions if show_suggestions else ""
                interpretation = self.current_interpretation if not (show_sarcasm or show_suggestions) else ""
                
                ui.draw_ai_coach_section(display_frame, interpretation, suggestions, sarcasm_alert)
                
                # Display frame
                cv2.imshow(window_name, display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key != 255:  # Any other key
                    self.handle_key_input(key)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Main loop error: {e}")
        finally:
            # Cleanup
            self.running = False
            self.vision_queue.put(None)  # Signal vision worker to stop
            
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final summary
            print("\n" + "="*50)
            print(self.session_analyzer.generate_summary())
            print("="*50)
            
            logger.info("Application terminated gracefully")


# ================================================================================================
# WEB APPLICATION - FastAPI-based interface
# ================================================================================================

if WEB_AVAILABLE:
    app = FastAPI(title="Autism Support System", description="Real-time Emotion Recognition for Autism Support")

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class ConnectionManager:
        def __init__(self):
            self.active_connections: List[WebSocket] = []
            self.system_components = None
            
        async def connect(self, websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)
            
            # Initialize system components for this connection
            if not self.system_components:
                self.system_components = {
                    'emotion_model': OptimizedCALMEDAutismCNN(),
                    'personalization': AdvancedPersonalizationEngine(),
                    'gemini_processor': GeminiLiveProcessor(),
                    'performance_monitor': PerformanceMonitor(),
                    'session_analyzer': SessionAnalyzer(),
                    'state': {
                        'current_emotion': 'Unknown',
                        'emotion_confidence': 0.0,
                        'monotony_score': 0.0,
                        'voice_stress': 0.0,
                        'transcribed_text': '',
                        'text_sentiment': 'Neutral',
                        'session_start': time.time(),
                        'last_ai_update': 0
                    }
                }
        
        def disconnect(self, websocket: WebSocket):
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        
        async def send_personal_message(self, message: dict, websocket: WebSocket):
            try:
                await websocket.send_json(message)
            except:
                await self.disconnect(websocket)
        
        async def broadcast(self, message: dict):
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except:
                    await self.disconnect(connection)

    manager = ConnectionManager()

    @app.get("/")
    async def get_homepage():
        """Serve the main autism-friendly interface"""
        return HTMLResponse(content=get_html_content(), status_code=200)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """Main WebSocket endpoint for real-time emotion analysis"""
        await manager.connect(websocket)
        
        try:
            while True:
                # Receive frame data from client
                data = await websocket.receive_json()
                
                if data.get('type') == 'video_frame':
                    # Process video frame
                    frame_data = data.get('frame')
                    if frame_data:
                        # Decode base64 frame
                        try:
                            frame_bytes = base64.b64decode(frame_data.split(',')[1])
                            nparr = np.frombuffer(frame_bytes, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            # Process emotion recognition
                            result = await process_emotion_frame(frame, manager.system_components)
                            
                            # Send results back to client
                            await manager.send_personal_message({
                                'type': 'emotion_result',
                                'data': result
                            }, websocket)
                        except Exception as e:
                            logger.error(f"Frame processing error: {e}")
                
                elif data.get('type') == 'request_help':
                    # Handle help request through Gemini API
                    help_response = await get_ai_suggestions(manager.system_components)
                    
                    await manager.send_personal_message({
                        'type': 'ai_coaching',
                        'subtype': 'suggestions',
                        'data': help_response
                    }, websocket)
                    
        except WebSocketDisconnect:
            manager.disconnect(websocket)
            logger.info("Client disconnected")

    async def process_emotion_frame(frame, components):
        """Process video frame for emotion recognition"""
        start_time = time.time()
        
        try:
            # Analyze emotion using the autism-specific CNN
            emotion_result = components['emotion_model'].analyze(frame)
            
            # Apply personalization
            final_emotion = components['personalization'].adapt_predictions(
                emotion_result, time.time()
            )
            
            # Update system state
            components['state']['current_emotion'] = final_emotion
            components['state']['emotion_confidence'] = emotion_result.get('confidence', 0.0)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            components['performance_monitor'].update(processing_time)
            performance_metrics = components['performance_monitor'].get_metrics()
            
            # Log emotion data
            components['session_analyzer'].log_emotion(emotion_result)
            
            # Get AI interpretation if enough time has passed
            current_time = time.time()
            ai_content = ""
            if current_time - components['state']['last_ai_update'] > 10:
                ai_content = components['gemini_processor'].get_emotion_interpretation(
                    final_emotion, 
                    components['state']['monotony_score']
                )
                if ai_content:
                    components['state']['last_ai_update'] = current_time
            
            return {
                'emotion': final_emotion,
                'confidence': emotion_result.get('confidence', 0.0),
                'emotion_scores': emotion_result.get('scores', {}),
                'processing_time_ms': processing_time * 1000,
                'performance': performance_metrics,
                'ai_interpretation': ai_content,
                'monotony_score': components['state']['monotony_score'],
                'voice_stress': components['state']['voice_stress'],
                'timestamp': datetime.now().strftime("%H:%M:%S")
            }
            
        except Exception as e:
            logger.error(f"Emotion processing error: {e}")
            return {
                'emotion': 'Error',
                'confidence': 0.0,
                'emotion_scores': {},
                'processing_time_ms': 0,
                'performance': {'status': 'ERROR'},
                'ai_interpretation': '',
                'monotony_score': 0.0,
                'voice_stress': 0.0,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            }

    async def get_ai_suggestions(components):
        """Get AI-powered suggestions through Gemini API"""
        try:
            suggestions = components['gemini_processor'].get_suggestions(
                components['state']['current_emotion'],
                f"confidence {components['state']['emotion_confidence']:.1%}"
            )
            return suggestions or "Try maintaining a calm and supportive environment."
        except Exception as e:
            logger.error(f"AI suggestions error: {e}")
            return "Unable to get suggestions at this time."

    def get_html_content():
        """Return the complete HTML content with autism-friendly design"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autism Support System</title>
    <style>
        /* Autism-friendly design principles */
        :root {
            --calm-blue: #E6F3FF;
            --soft-green: #F0F8F0;
            --neutral-gray: #F5F5F5;
            --accent-blue: #4A90E2;
            --text-dark: #2C3E50;
            --success-green: #27AE60;
            --warning-orange: #F39C12;
            --error-red: #E74C3C;
            --border-light: #E0E0E0;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, var(--calm-blue) 0%, var(--soft-green) 100%);
            color: var(--text-dark);
            line-height: 1.6;
            min-height: 100vh;
        }

        .main-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            display: grid;
            grid-template-columns: 1fr 450px;
            gap: 30px;
            min-height: 100vh;
        }

        /* Video Section - Clean and Simple */
        .video-section {
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 2px solid var(--border-light);
        }

        .video-container {
            position: relative;
            width: 100%;
            max-width: 640px;
            margin: 0 auto;
            background: var(--neutral-gray);
            border-radius: 12px;
            overflow: hidden;
            border: 3px solid var(--border-light);
        }

        #videoElement {
            width: 100%;
            height: auto;
            display: block;
        }

        .video-overlay {
            position: absolute;
            top: 15px;
            left: 15px;
            background: rgba(255,255,255,0.95);
            padding: 12px 18px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 18px;
            border: 2px solid var(--accent-blue);
            min-width: 200px;
            text-align: center;
        }

        /* Controls Section */
        .controls-section {
            margin-top: 25px;
            text-align: center;
        }

        .control-button {
            background: var(--accent-blue);
            color: white;
            border: none;
            padding: 14px 28px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            margin: 0 10px;
            transition: all 0.3s ease;
            min-width: 140px;
        }

        .control-button:hover:not(:disabled) {
            background: #357ABD;
            transform: translateY(-2px);
        }

        .control-button:disabled {
            background: #BDC3C7;
            cursor: not-allowed;
            transform: none;
        }

        .help-button {
            background: var(--warning-orange);
            margin-top: 15px;
        }

        .help-button:hover:not(:disabled) {
            background: #D68910;
        }

        /* Analysis Panel - Autism-Friendly Layout */
        .analysis-panel {
            background: white;
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 2px solid var(--border-light);
            display: flex;
            flex-direction: column;
            gap: 20px;
            height: fit-content;
        }

        .panel-header {
            text-align: center;
            padding: 18px;
            background: var(--calm-blue);
            border-radius: 12px;
            border: 2px solid var(--accent-blue);
        }

        .panel-title {
            font-size: 22px;
            font-weight: 700;
            color: var(--text-dark);
            margin-bottom: 8px;
        }

        .status-indicator {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            font-size: 16px;
        }

        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--success-green);
        }

        /* Performance Metrics */
        .performance-section {
            background: var(--neutral-gray);
            padding: 18px;
            border-radius: 12px;
            border: 2px solid var(--border-light);
        }

        .section-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 12px;
            text-align: center;
            color: var(--text-dark);
        }

        .performance-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }

        .metric-item {
            text-align: center;
            padding: 12px;
            background: white;
            border-radius: 8px;
            border: 1px solid var(--border-light);
        }

        .metric-label {
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }

        .metric-value {
            font-size: 18px;
            font-weight: 700;
        }

        /* Emotion Analysis */
        .emotion-section {
            background: var(--soft-green);
            padding: 20px;
            border-radius: 12px;
            border: 2px solid var(--success-green);
        }

        .current-emotion {
            text-align: center;
            margin-bottom: 18px;
        }

        .emotion-display {
            font-size: 28px;
            font-weight: 800;
            color: var(--text-dark);
            margin-bottom: 8px;
        }

        .confidence-display {
            font-size: 16px;
            color: #666;
        }

        .emotion-breakdown {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .emotion-bar {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .emotion-label {
            min-width: 75px;
            font-size: 14px;
            font-weight: 600;
            text-transform: capitalize;
        }

        .bar-container {
            flex: 1;
            height: 18px;
            background: white;
            border-radius: 9px;
            overflow: hidden;
            border: 1px solid var(--border-light);
        }

        .bar-fill {
            height: 100%;
            background: var(--accent-blue);
            transition: width 0.5s ease;
        }

        .emotion-value {
            min-width: 35px;
            text-align: right;
            font-size: 12px;
            font-weight: 600;
        }

        /* Audio Analysis */
        .audio-section {
            background: var(--calm-blue);
            padding: 18px;
            border-radius: 12px;
            border: 2px solid var(--accent-blue);
        }

        .audio-metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }

        .audio-metric {
            text-align: center;
            padding: 15px;
            background: white;
            border-radius: 8px;
            border: 1px solid var(--border-light);
        }

        /* AI Coaching Section */
        .ai-section {
            background: linear-gradient(135deg, #FFF9E6 0%, #FFF3CD 100%);
            padding: 20px;
            border-radius: 12px;
            border: 2px solid var(--warning-orange);
            min-height: 120px;
        }

        .ai-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 15px;
        }

        .ai-icon {
            width: 28px;
            height: 28px;
            background: var(--warning-orange);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 14px;
        }

        .ai-title {
            font-size: 18px;
            font-weight: 700;
            color: var(--text-dark);
        }

        .ai-content {
            font-size: 16px;
            line-height: 1.5;
            color: var(--text-dark);
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid var(--warning-orange);
            min-height: 60px;
        }

        /* Connection Status */
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            padding: 12px 18px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border: 2px solid var(--border-light);
            z-index: 1000;
            font-weight: 600;
        }

        .status-connected {
            border-color: var(--success-green);
            color: var(--success-green);
        }

        .status-disconnected {
            border-color: var(--error-red);
            color: var(--error-red);
        }

        /* Responsive Design */
        @media (max-width: 1024px) {
            .main-container {
                grid-template-columns: 1fr;
                gap: 20px;
            }
            
            .analysis-panel {
                order: -1;
            }
        }

        @media (max-width: 768px) {
            .main-container {
                padding: 15px;
            }
            
            .performance-grid,
            .audio-metrics {
                grid-template-columns: 1fr;
            }
        }

        /* Focus styles for keyboard navigation */
        button:focus,
        .control-button:focus {
            outline: 3px solid var(--accent-blue);
            outline-offset: 2px;
        }

        /* Loading animation */
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .loading {
            animation: pulse 1.5s infinite;
        }
    </style>
</head>
<body>
    <div class="connection-status" id="connectionStatus">
        <span>🔴 Disconnected</span>
    </div>

    <div class="main-container">
        <!-- Video Section -->
        <div class="video-section">
            <h1 style="text-align: center; font-size: 28px; margin-bottom: 25px; color: var(--text-dark);">
                Autism Support System
            </h1>
            
            <div class="video-container">
                <video id="videoElement" autoplay muted playsinline></video>
                <div class="video-overlay" id="emotionOverlay">
                    <span id="currentEmotion">Initializing...</span>
                </div>
            </div>
            
            <div class="controls-section">
                <button class="control-button" id="startButton">Start Camera</button>
                <button class="control-button" id="stopButton" disabled>Stop Camera</button>
                <br>
                <button class="control-button help-button" id="helpButton" disabled>Get Suggestions</button>
            </div>
        </div>

        <!-- Analysis Panel -->
        <div class="analysis-panel">
            <!-- Panel Header -->
            <div class="panel-header">
                <div class="panel-title">Real-Time Analysis</div>
                <div class="status-indicator">
                    <div class="status-dot" id="systemStatus"></div>
                    <span id="statusText">System Ready</span>
                </div>
            </div>

            <!-- Performance Metrics -->
            <div class="performance-section">
                <div class="section-title">System Performance</div>
                <div class="performance-grid">
                    <div class="metric-item">
                        <div class="metric-label">FPS</div>
                        <div class="metric-value" id="fpsDisplay">--</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Latency</div>
                        <div class="metric-value" id="latencyDisplay">-- ms</div>
                    </div>
                </div>
            </div>

            <!-- Emotion Analysis -->
            <div class="emotion-section">
                <div class="section-title">Emotion Recognition</div>
                <div class="current-emotion">
                    <div class="emotion-display" id="mainEmotion">Unknown</div>
                    <div class="confidence-display" id="confidenceLevel">0% confidence</div>
                </div>
                <div class="emotion-breakdown" id="emotionBreakdown">
                    <!-- Emotion bars will be populated dynamically -->
                </div>
            </div>

            <!-- Audio Analysis -->
            <div class="audio-section">
                <div class="section-title">Audio Analysis</div>
                <div class="audio-metrics">
                    <div class="audio-metric">
                        <div class="metric-label">Speech Monotony</div>
                        <div class="metric-value" id="monotonyScore">--</div>
                    </div>
                    <div class="audio-metric">
                        <div class="metric-label">Voice Stress</div>
                        <div class="metric-value" id="voiceStress">--</div>
                    </div>
                </div>
            </div>

            <!-- AI Coaching -->
            <div class="ai-section">
                <div class="ai-header">
                    <div class="ai-icon">AI</div>
                    <div class="ai-title">Support Coach</div>
                </div>
                <div class="ai-content" id="aiContent">
                    Welcome! Start the camera to begin real-time emotion analysis. I'll provide insights and suggestions to support positive interactions.
                </div>
            </div>
        </div>
    </div>

    <script>
        class AutismSupportSystem {
            constructor() {
                this.websocket = null;
                this.videoElement = document.getElementById('videoElement');
                this.canvas = document.createElement('canvas');
                this.context = this.canvas.getContext('2d');
                this.isStreaming = false;
                this.frameInterval = null;
                
                this.initializeElements();
                this.setupEventListeners();
                this.connectWebSocket();
            }

            initializeElements() {
                this.elements = {
                    startButton: document.getElementById('startButton'),
                    stopButton: document.getElementById('stopButton'),
                    helpButton: document.getElementById('helpButton'),
                    connectionStatus: document.getElementById('connectionStatus'),
                    currentEmotion: document.getElementById('currentEmotion'),
                    emotionOverlay: document.getElementById('emotionOverlay'),
                    mainEmotion: document.getElementById('mainEmotion'),
                    confidenceLevel: document.getElementById('confidenceLevel'),
                    emotionBreakdown: document.getElementById('emotionBreakdown'),
                    fpsDisplay: document.getElementById('fpsDisplay'),
                    latencyDisplay: document.getElementById('latencyDisplay'),
                    monotonyScore: document.getElementById('monotonyScore'),
                    voiceStress: document.getElementById('voiceStress'),
                    aiContent: document.getElementById('aiContent'),
                    systemStatus: document.getElementById('systemStatus'),
                    statusText: document.getElementById('statusText')
                };
            }

            setupEventListeners() {
                this.elements.startButton.addEventListener('click', () => this.startCamera());
                this.elements.stopButton.addEventListener('click', () => this.stopCamera());
                this.elements.helpButton.addEventListener('click', () => this.requestHelp());
                
                // Keyboard shortcuts for accessibility
                document.addEventListener('keydown', (e) => {
                    if (e.altKey && e.key === 's') {
                        e.preventDefault();
                        if (!this.elements.startButton.disabled) {
                            this.startCamera();
                        }
                    } else if (e.altKey && e.key === 'h') {
                        e.preventDefault();
                        if (!this.elements.helpButton.disabled) {
                            this.requestHelp();
                        }
                    }
                });
            }

            connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                
                this.websocket = new WebSocket(wsUrl);
                
                this.websocket.onopen = () => {
                    console.log('WebSocket connected');
                    this.updateConnectionStatus(true);
                };
                
                this.websocket.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                };
                
                this.websocket.onclose = () => {
                    console.log('WebSocket disconnected');
                    this.updateConnectionStatus(false);
                    // Auto-reconnect after 3 seconds
                    setTimeout(() => this.connectWebSocket(), 3000);
                };
                
                this.websocket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.updateConnectionStatus(false);
                };
            }

            updateConnectionStatus(connected) {
                const status = this.elements.connectionStatus;
                if (connected) {
                    status.innerHTML = '<span>🟢 Connected</span>';
                    status.className = 'connection-status status-connected';
                    this.elements.startButton.disabled = false;
                } else {
                    status.innerHTML = '<span>🔴 Disconnected</span>';
                    status.className = 'connection-status status-disconnected';
                    this.elements.startButton.disabled = true;
                    this.elements.helpButton.disabled = true;
                }
            }

            async startCamera() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({
                        video: { width: 640, height: 480 },
                        audio: false  // Audio processing handled by backend
                    });
                    
                    this.videoElement.srcObject = stream;
                    this.isStreaming = true;
                    
                    this.elements.startButton.disabled = true;
                    this.elements.stopButton.disabled = false;
                    this.elements.helpButton.disabled = false;
                    
                    // Start sending frames
                    this.startFrameCapture();
                    
                    this.updateSystemStatus('active', 'Analyzing...');
                    
                } catch (error) {
                    console.error('Error accessing camera:', error);
                    this.showError('Camera access denied. Please check permissions.');
                }
            }

            stopCamera() {
                if (this.videoElement.srcObject) {
                    const tracks = this.videoElement.srcObject.getTracks();
                    tracks.forEach(track => track.stop());
                    this.videoElement.srcObject = null;
                }
                
                if (this.frameInterval) {
                    clearInterval(this.frameInterval);
                    this.frameInterval = null;
                }
                
                this.isStreaming = false;
                
                this.elements.startButton.disabled = false;
                this.elements.stopButton.disabled = true;
                this.elements.helpButton.disabled = true;
                
                this.updateSystemStatus('ready', 'System Ready');
                this.resetDisplays();
            }

            startFrameCapture() {
                // Capture and send frames at 8 FPS for optimal performance
                this.frameInterval = setInterval(() => {
                    if (this.isStreaming && this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                        this.captureAndSendFrame();
                    }
                }, 125); // 8 FPS
            }

            captureAndSendFrame() {
                // Set canvas size to match video
                this.canvas.width = this.videoElement.videoWidth || 640;
                this.canvas.height = this.videoElement.videoHeight || 480;
                
                // Draw video frame to canvas
                this.context.drawImage(this.videoElement, 0, 0);
                
                // Convert to base64 and send
                const frameData = this.canvas.toDataURL('image/jpeg', 0.7);
                
                this.websocket.send(JSON.stringify({
                    type: 'video_frame',
                    frame: frameData
                }));
            }

            handleWebSocketMessage(data) {
                switch (data.type) {
                    case 'emotion_result':
                        this.updateEmotionDisplay(data.data);
                        break;
                    case 'ai_coaching':
                        this.updateAICoaching(data);
                        break;
                    default:
                        console.log('Unknown message type:', data.type);
                }
            }

            updateEmotionDisplay(emotionData) {
                const emotion = emotionData.emotion || 'Unknown';
                const confidence = (emotionData.confidence * 100) || 0;
                
                // Update main displays
                this.elements.currentEmotion.textContent = emotion;
                this.elements.mainEmotion.textContent = emotion;
                this.elements.confidenceLevel.textContent = `${confidence.toFixed(0)}% confidence`;
                
                // Update emotion breakdown bars
                this.updateEmotionBars(emotionData.emotion_scores || {});
                
                // Update performance metrics
                if (emotionData.performance) {
                    this.elements.fpsDisplay.textContent = emotionData.performance.fps?.toFixed(1) || '--';
                    this.elements.latencyDisplay.textContent = 
                        emotionData.processing_time_ms ? `${emotionData.processing_time_ms.toFixed(0)} ms` : '-- ms';
                }
                
                // Update audio metrics
                this.elements.monotonyScore.textContent = 
                    emotionData.monotony_score ? emotionData.monotony_score.toFixed(2) : '--';
                this.elements.voiceStress.textContent = 
                    emotionData.voice_stress ? emotionData.voice_stress.toFixed(2) : '--';
                
                // Update AI interpretation if available
                if (emotionData.ai_interpretation) {
                    this.elements.aiContent.textContent = emotionData.ai_interpretation;
                }
                
                // Update system status based on performance
                if (emotionData.performance?.status === 'OPTIMAL') {
                    this.updateSystemStatus('optimal', 'Optimal Performance');
                } else if (emotionData.performance?.status === 'SLOW') {
                    this.updateSystemStatus('warning', 'Performance Issues');
                }
            }

            updateEmotionBars(emotionScores) {
                const container = this.elements.emotionBreakdown;
                container.innerHTML = '';
                
                // Sort emotions by score
                const sortedEmotions = Object.entries(emotionScores)
                    .sort(([,a], [,b]) => b - a)
                    .slice(0, 5); // Show top 5 emotions
                
                sortedEmotions.forEach(([emotion, score]) => {
                    const barElement = document.createElement('div');
                    barElement.className = 'emotion-bar';
                    barElement.innerHTML = `
                        <div class="emotion-label">${emotion}</div>
                        <div class="bar-container">
                            <div class="bar-fill" style="width: ${score}%"></div>
                        </div>
                        <div class="emotion-value">${score.toFixed(0)}%</div>
                    `;
                    container.appendChild(barElement);
                });
            }

            updateAICoaching(coachingData) {
                if (coachingData.subtype === 'suggestions') {
                    this.elements.aiContent.textContent = coachingData.data || 'No suggestions available.';
                    
                    // Highlight the AI section briefly
                    const aiSection = this.elements.aiContent.parentElement;
                    aiSection.style.transform = 'scale(1.02)';
                    aiSection.style.transition = 'transform 0.3s ease';
                    setTimeout(() => {
                        aiSection.style.transform = 'scale(1)';
                    }, 300);
                }
            }

            updateSystemStatus(status, text) {
                const statusDot = this.elements.systemStatus;
                const statusText = this.elements.statusText;
                
                statusText.textContent = text;
                
                switch (status) {
                    case 'ready':
                        statusDot.style.background = '#BDC3C7';
                        break;
                    case 'active':
                        statusDot.style.background = '#3498DB';
                        break;
                    case 'optimal':
                        statusDot.style.background = '#27AE60';
                        break;
                    case 'warning':
                        statusDot.style.background = '#F39C12';
                        break;
                    case 'error':
                        statusDot.style.background = '#E74C3C';
                        break;
                }
            }

            requestHelp() {
                if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                    this.websocket.send(JSON.stringify({
                        type: 'request_help'
                    }));
                    
                    // Provide immediate feedback
                    this.elements.aiContent.textContent = 'Getting personalized suggestions...';
                    this.elements.aiContent.classList.add('loading');
                    
                    // Remove loading animation after 3 seconds
                    setTimeout(() => {
                        this.elements.aiContent.classList.remove('loading');
                    }, 3000);
                }
            }

            showError(message) {
                this.elements.aiContent.textContent = `⚠️ ${message}`;
                this.updateSystemStatus('error', 'Error');
            }

            resetDisplays() {
                this.elements.currentEmotion.textContent = 'Not Active';
                this.elements.mainEmotion.textContent = 'Unknown';
                this.elements.confidenceLevel.textContent = '0% confidence';
                this.elements.fpsDisplay.textContent = '--';
                this.elements.latencyDisplay.textContent = '-- ms';
                this.elements.monotonyScore.textContent = '--';
                this.elements.voiceStress.textContent = '--';
                this.elements.emotionBreakdown.innerHTML = '';
                this.elements.aiContent.textContent = 'System stopped. Start camera to begin analysis.';
            }
        }

        // Initialize the system when the page loads
        document.addEventListener('DOMContentLoaded', () => {
            console.log('Initializing Autism Support System...');
            new AutismSupportSystem();
        });
    </script>
</body>
</html>
        """

else:
    # Fallback if FastAPI is not available
    def run_web_server():
        print("FastAPI not available. Please install with: pip install fastapi uvicorn websockets")
        return


# ================================================================================================
# MAIN APPLICATION ENTRY POINT
# ================================================================================================

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description='Autism Support System - Complete Emotion Recognition Platform')
    parser.add_argument('--mode', choices=['desktop', 'web'], default='desktop',
                        help='Run mode: desktop (OpenCV GUI) or web (FastAPI server)')
    parser.add_argument('--host', default='0.0.0.0', help='Web server host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Web server port (default: 8000)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("    AUTISM SUPPORT SYSTEM - EMOTION RECOGNITION PLATFORM")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Features Available:")
    print(f"  - DeepFace Emotion Recognition: {'✓' if DEEPFACE_AVAILABLE else '✗'}")
    print(f"  - Audio Processing: {'✓' if AUDIO_AVAILABLE else '✗'}")
    print(f"  - Speech Recognition: {'✓' if SPEECH_AVAILABLE else '✗'}")
    print(f"  - Gemini AI Integration: {'✓' if GEMINI_AVAILABLE else '✗'}")
    if args.mode == 'web':
        print(f"  - Web Interface: {'✓' if WEB_AVAILABLE else '✗'}")
    print("=" * 60)
    
    if args.mode == 'desktop':
        # Run desktop application
        try:
            system = OptimizedAutismEmotionSystem()
            system.main_loop()
        except KeyboardInterrupt:
            print("\nDesktop application terminated by user.")
        except Exception as e:
            logger.error(f"Desktop application error: {e}")
            print(f"Error: {e}")
    
    elif args.mode == 'web':
        # Run web application
        if WEB_AVAILABLE:
            try:
                print(f"\nStarting web server...")
                print(f"URL: http://{args.host}:{args.port}")
                print("\nWeb Controls:")
                print("- Start Camera: Begin emotion analysis")
                print("- Get Suggestions: Request AI coaching")
                print("- Alt+S: Quick start camera")
                print("- Alt+H: Quick help request")
                print("\nPress Ctrl+C to stop the server")
                
                import uvicorn
                uvicorn.run(app, host=args.host, port=args.port, log_level="info")
            except KeyboardInterrupt:
                print("\nWeb server terminated by user.")
            except Exception as e:
                logger.error(f"Web server error: {e}")
                print(f"Error: {e}")
        else:
            print("Web mode not available. Please install FastAPI and dependencies:")
            print("pip install fastapi uvicorn websockets python-multipart")


if __name__ == '__main__':
    main()
