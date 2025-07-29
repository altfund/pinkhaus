"""
Speaker diarization module for integrating pyannote-audio with the transcription pipeline.

This module provides speaker identification capabilities to enhance transcription
with "who speaks when" information.
"""

import os
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import torch
import torchaudio
from huggingface_hub import hf_hub_download, HfApi


@dataclass
class SpeakerSegment:
    """Represents a segment of audio attributed to a specific speaker."""
    start: float  # Start time in seconds
    end: float    # End time in seconds
    speaker: str  # Speaker label (e.g., "SPEAKER_0", "SPEAKER_1")
    confidence: Optional[float] = None  # Optional confidence score


@dataclass
class DiarizationResult:
    """Result of speaker diarization process."""
    segments: List[SpeakerSegment]
    num_speakers: int
    audio_duration: float


class SpeakerDiarizer:
    """
    Speaker diarization using pyannote-audio models from Hugging Face.
    
    This class provides a lightweight interface for speaker diarization
    that can work with or without the full pyannote-audio library.
    """
    
    def __init__(self, auth_token: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the speaker diarizer.
        
        Args:
            auth_token: Hugging Face authentication token (required for some models)
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detect)
        """
        self.auth_token = auth_token or os.getenv("HF_TOKEN")
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._pipeline = None
        
    def _load_pipeline(self):
        """Load the diarization pipeline lazily."""
        if self._pipeline is None:
            try:
                # Try to import pyannote if available
                from pyannote.audio import Pipeline
                
                # Load pretrained pipeline from Hugging Face
                model_name = "pyannote/speaker-diarization-3.1"
                self._pipeline = Pipeline.from_pretrained(
                    model_name,
                    use_auth_token=self.auth_token
                )
                self._pipeline.to(torch.device(self.device))
                
            except ImportError:
                raise ImportError(
                    "pyannote-audio is not installed. Due to dependency conflicts "
                    "with Python 3.13, you may need to use Python 3.11 or wait for "
                    "updated packages. Alternative: use the mock diarization mode."
                )
    
    def diarize_file(
        self, 
        audio_path: str, 
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        use_mock: bool = False
    ) -> DiarizationResult:
        """
        Perform speaker diarization on an audio file.
        
        Args:
            audio_path: Path to the audio file
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)
            use_mock: Use mock diarization for testing/demo purposes
            
        Returns:
            DiarizationResult with speaker segments
        """
        if use_mock:
            return self._mock_diarization(audio_path)
        
        # Load pipeline if needed
        self._load_pipeline()
        
        # Run diarization
        diarization = self._pipeline(
            audio_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        # Convert to our format
        segments = []
        speakers = set()
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(SpeakerSegment(
                start=turn.start,
                end=turn.end,
                speaker=speaker
            ))
            speakers.add(speaker)
        
        # Get audio duration
        waveform, sample_rate = torchaudio.load(audio_path)
        duration = waveform.shape[1] / sample_rate
        
        return DiarizationResult(
            segments=segments,
            num_speakers=len(speakers),
            audio_duration=duration
        )
    
    def _mock_diarization(self, audio_path: str) -> DiarizationResult:
        """
        Generate mock diarization results for testing/demo.
        
        This simulates a conversation between two speakers.
        """
        # Get audio duration
        waveform, sample_rate = torchaudio.load(audio_path)
        duration = waveform.shape[1] / sample_rate
        
        # Generate mock speaker segments
        segments = []
        current_time = 0.0
        speaker_idx = 0
        
        while current_time < duration:
            # Simulate speaker turns of 5-15 seconds
            segment_duration = min(5 + torch.rand(1).item() * 10, duration - current_time)
            
            segments.append(SpeakerSegment(
                start=current_time,
                end=current_time + segment_duration,
                speaker=f"SPEAKER_{speaker_idx}",
                confidence=0.85 + torch.rand(1).item() * 0.15  # Mock confidence
            ))
            
            current_time += segment_duration
            speaker_idx = 1 - speaker_idx  # Alternate between 2 speakers
        
        return DiarizationResult(
            segments=segments,
            num_speakers=2,
            audio_duration=duration
        )
    
    def align_with_transcription(
        self,
        diarization: DiarizationResult,
        transcription_segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Align speaker diarization with transcription segments.
        
        Args:
            diarization: Speaker diarization result
            transcription_segments: List of transcription segments with start/end times
            
        Returns:
            Enhanced transcription segments with speaker information
        """
        enhanced_segments = []
        
        for trans_seg in transcription_segments:
            # Convert times to float if needed
            if isinstance(trans_seg.get('start'), str):
                # Parse time string if needed
                start = self._parse_time(trans_seg['start'])
                end = self._parse_time(trans_seg['end'])
            else:
                start = float(trans_seg.get('start', 0))
                end = float(trans_seg.get('end', 0))
            
            # Find overlapping speaker segments
            overlapping_speakers = []
            for speaker_seg in diarization.segments:
                # Check if segments overlap
                if (speaker_seg.start < end and speaker_seg.end > start):
                    overlap_duration = min(speaker_seg.end, end) - max(speaker_seg.start, start)
                    overlapping_speakers.append((speaker_seg.speaker, overlap_duration))
            
            # Assign speaker based on maximum overlap
            if overlapping_speakers:
                assigned_speaker = max(overlapping_speakers, key=lambda x: x[1])[0]
            else:
                assigned_speaker = "UNKNOWN"
            
            # Create enhanced segment
            enhanced_seg = trans_seg.copy()
            enhanced_seg['speaker'] = assigned_speaker
            enhanced_segments.append(enhanced_seg)
        
        return enhanced_segments
    
    def _parse_time(self, time_str: str) -> float:
        """Parse time string in HH:MM:SS.mmm format to seconds."""
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds


def create_speaker_aware_transcription(
    audio_path: str,
    transcription_result: Any,
    hf_token: Optional[str] = None,
    use_mock: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to add speaker diarization to existing transcription.
    
    Args:
        audio_path: Path to the audio file
        transcription_result: Existing transcription result
        hf_token: Hugging Face token for model access
        use_mock: Use mock diarization for testing
        
    Returns:
        Enhanced transcription with speaker information
    """
    # Initialize diarizer
    diarizer = SpeakerDiarizer(auth_token=hf_token)
    
    # Perform diarization
    diarization = diarizer.diarize_file(audio_path, use_mock=use_mock)
    
    # Get segments from transcription result
    if hasattr(transcription_result, 'to_dict'):
        trans_dict = transcription_result.to_dict()
        segments = trans_dict.get('segments', [])
    else:
        segments = transcription_result.get('segments', [])
    
    # Align speakers with transcription
    enhanced_segments = diarizer.align_with_transcription(diarization, segments)
    
    # Create enhanced result
    enhanced_result = trans_dict.copy() if hasattr(transcription_result, 'to_dict') else transcription_result.copy()
    enhanced_result['segments'] = enhanced_segments
    enhanced_result['num_speakers'] = diarization.num_speakers
    enhanced_result['has_speaker_labels'] = True
    
    return enhanced_result