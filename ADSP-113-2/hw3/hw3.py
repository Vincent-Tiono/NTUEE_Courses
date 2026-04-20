import sounddevice as sd
import soundfile as sf
import numpy as np

def play_audio(audio_data, sample_rate=8192):
    """
    Plays the audio signal at the specified sample rate.

    Args:
        audio_data (array_like): Audio data.
        sample_rate (int, optional): Sample rate in Hz. Defaults to 8192.
    """
    audio_data = np.asarray(audio_data)
    sd.play(audio_data, samplerate=sample_rate)
    sd.wait()  # Wait until playing is finished

def calculate_frequency(notation: str, reference_freq: float):
    """
    Calculate the frequency of a note based on notation and reference frequency.
    
    Args:
        notation (str): Musical notation ('1'-'7', can include modifiers)
        reference_freq (float): Reference frequency (C4 = 261.63 Hz by default)
    
    Returns:
        float: Frequency in Hz
    """
    # Handle rests
    if notation == '0':
        return 0
    
    # Define frequency ratios for notes
    frequency_map = {
        1: reference_freq,  # Do (C)
        2: reference_freq * (2**(2/12)),  # Re (D)
        3: reference_freq * (2**(4/12)),  # Mi (E)
        4: reference_freq * (2**(5/12)),  # Fa (F)
        5: reference_freq * (2**(7/12)),  # Sol (G)
        6: reference_freq * (2**(9/12)),  # La (A)
        7: reference_freq * (2**(11/12)),  # Ti (B)
    }
    
    # Define frequency multipliers for modifiers
    modifier_map = {
        "#": 2**(1/12),   # Sharp (semitone up)
        "b": 2**(-1/12),  # Flat (semitone down)
        "^": 2,           # Octave up
        "v": 0.5,         # Octave down
    }
    
    # Count octave shifts
    octave_up_count = notation.count('^')
    octave_down_count = notation.count('v')
    
    # Extract base note by removing modifiers
    clean_notation = notation
    for symbol in ['^', 'v', '#', 'b']:
        clean_notation = clean_notation.replace(symbol, '')
    
    # Check for accidentals
    is_sharp = '#' in notation
    is_flat = 'b' in notation
    
    # Calculate base frequency
    frequency = frequency_map[int(clean_notation)]
    
    # Apply accidentals
    if is_sharp:
        frequency *= modifier_map['#']
    if is_flat:
        frequency *= modifier_map['b']
    
    # Apply octave shifts
    if octave_up_count > 0:
        frequency *= (modifier_map['^'] ** octave_up_count)
    if octave_down_count > 0:
        frequency *= (modifier_map['v'] ** octave_down_count)
    
    return frequency

def generate_music(notation_sequence: list, duration_sequence: list, output_filename, tempo, reference_freq=261.63, volume=1):
    """
    Generate music from numeric notation and duration values.

    Args:
        notation_sequence (list): List of musical notations in numeric format
        duration_sequence (list): List of relative note durations
        output_filename (str): Name of output file (without extension)
        tempo (int): Tempo in beats per minute (BPM)
        reference_freq (float): Reference frequency in Hz, default is 261.63 Hz (C4)
        volume (float): Volume level between 0.1 and 5

    Features:
    1. Adjustable volume levels
    2. Rich harmonics for fuller sound
    3. Flexible tempo control (BPM)
    4. Customizable key signatures via reference frequency
    5. Support for accidentals (sharp/flat notes)
    6. Multiple octave shifts: each '^' adds one octave up, each 'v' adds one octave down
       (e.g., '1^' is one octave up, '1^^' is two octaves up, '2v' is one octave down)
    7. Rest notation: use '0' to represent silence
    8. Decay envelope for smoother transitions between notes
    9. Combined modifiers: accidentals and octave shifts can be combined (e.g., '#1^')
    10. Automatic normalization to prevent audio clipping

    Note: Use '0' for rests (silence) and multiple '^' or 'v' for octave shifts
    """
    # Validate and adjust volume
    if volume > 5:
        print("Volume too high, setting to 5")
        volume = 5
    elif volume < 0.1:
        print("Volume too low, setting to 0.1")
        volume = 0.1
    
    # Audio settings
    sample_rate = 11025  # Sample rate in Hz
    beat_duration = 60/tempo  # Duration of one beat in seconds
    audio_segments = []
    
    # Generate each note
    for i in range(len(notation_sequence)):
        note_duration = beat_duration * duration_sequence[i]
        
        if notation_sequence[i] == '0':
            # Generate silence for rest
            segment = np.zeros(int(sample_rate * note_duration))
        else:
            # Calculate note frequency
            frequency = calculate_frequency(notation_sequence[i], reference_freq)
            
            # Create time array
            time_array = np.linspace(0, note_duration, int(sample_rate * note_duration), endpoint=False)
            
            # Generate waveform with harmonics
            segment = (
                volume * np.cos(2 * np.pi * frequency * time_array) +  # Fundamental frequency
                volume * 0.3 * np.cos(2 * np.pi * 2 * frequency * time_array) +  # First harmonic
                volume * 0.05 * np.cos(2 * np.pi * 4 * frequency * time_array)   # Second harmonic
            )
            
            # Apply decay envelope
            decay_duration = 0.2 * note_duration
            decay_start = int(sample_rate * (note_duration - decay_duration))
            decay_end = int(sample_rate * note_duration)
            decay_curve = np.linspace(1.0, 0.0, decay_end - decay_start)
            segment[decay_start:decay_end] *= decay_curve
            
            # Normalize to prevent clipping
            segment /= np.max(np.abs(segment))
            
        audio_segments.append(segment)
    
    # Combine all audio segments
    complete_audio = np.concatenate(audio_segments)
    
    # Save to file
    print(f"Generated {output_filename}.wav with BPM: {tempo}")
    # play_audio(complete_audio, sample_rate)  # Uncomment to play audio
    sf.write(f'{output_filename}.wav', complete_audio, sample_rate)

# Turkish March segments
phrase_1 = ['7', '6', '#5', '6', '1^', '0']
timing_1 = [0.25, 0.25, 0.25, 0.25, 0.5, 0.5]

phrase_2 = ['2^', '1^', '7', '1^', '3^', '0']
timing_2 = [0.25, 0.25, 0.25, 0.25, 0.5, 0.5]

phrase_3 = ['4^', '3^', '#2^', '3^', '7^', '6^', '#5^', '6^', '7^', '6^', '#5^', '6^', '1^^']
timing_3 = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 1]

phrase_4 = ['6^', '1^^', '5^', '6^', '7^', '6^', '5^', '6^']
timing_4 = [0.5, 0.5, 0.0625, 0.0625, 0.5, 0.5, 0.5, 0.5]

phrase_5 = ['5^', '6^', '7^', '6^', '5^', '6^']
timing_5 = [0.0625, 0.0625, 0.5, 0.5, 0.5, 0.5]

phrase_6 = ['5^', '6^', '7^', '6^', '5^', '#4^', '3^']
timing_6 = [0.0625, 0.0625, 0.5, 0.5, 0.5, 0.5, 0.5]

# Combine all phrases
complete_notation = phrase_1 + phrase_2 + phrase_3 + phrase_4 + phrase_5 + phrase_6
complete_timing = timing_1 + timing_2 + timing_3 + timing_4 + timing_5 + timing_6
song_name = "turkish_march"

# Generate music at 160 BPM using D4 as reference frequency
generate_music(complete_notation, complete_timing, song_name, 160, 293.66, volume=0.2)