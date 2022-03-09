import pyaudio
import wave
import os
import random
from teste_pi import compare_file

form_1 = pyaudio.paInt16 # 16-bit resolution
chans = 1 # 1 channel
samp_rate = 44100 # 44.1kHz sampling rate
chunk = 4096 # 2^12 samples for buffer
record_secs = 5 # seconds to record
dev_index = 1 # device index found by p.get_device_info_by_index(ii)
wav_output_filename = 'output.wav' # name of .wav file
wav_introduction = 'introduction.wav' #introduction audio
wav_test_note = ['la.wav','sib.wav','si.wav','do.wav','reb.wav','re.wav','mib.wav','mi.wav','fa.wav','fas','sol.wav','lab.wav'] #test note
freq_test_note = [440.0, 466.16, 493.88, 523.25, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.3] #notes frequency Hz
wav_correct = 'test_correct.wav'
wav_incorrect = 'test_incorrect.wav'

audio = pyaudio.PyAudio() # create pyaudio instantiation

index = random.randint(0,11)
test_note = wav_test_note[index]
test_freq = freq_test_note[index]

os.system('aplay introduction.wav')
os.system('aplay '+test_note)
# create pyaudio stream
stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                    input_device_index = dev_index,input = True, \
                    frames_per_buffer=chunk)
print("recording")
frames = []

# loop through stream and append audio chunks to frame array
for ii in range(0,int((samp_rate/chunk)*record_secs)):
    data = stream.read(chunk)
    frames.append(data)

print("finished recording")

# stop the stream, close it, and terminate the pyaudio instantiation
stream.stop_stream()
stream.close()
audio.terminate()

# save the audio frames as .wav file
wavefile = wave.open(wav_output_filename,'wb')
wavefile.setnchannels(chans)
wavefile.setsampwidth(audio.get_sample_size(form_1))
wavefile.setframerate(samp_rate)
wavefile.writeframes(b''.join(frames))
wavefile.close()

os.system('aplay output.wav')

result = compare_file(test_freq,'output.wav')

if result: os.system('aplay test_correct.wav')
else: os.system('aplay test_incorrect.wav')