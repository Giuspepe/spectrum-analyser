import numpy as np

chunk_size = 4096  # Use a multiple of 8 (FFT will compute faster with a multiple of 8)
sampling_frequency = 44100  # Hz
channels = 2  # TODO: change to 1 when using mono sound card

frequency_bins = [(0, 156), (156, 313), (313, 625), (625, 1250), (1250, 2500),
                  (2500, 5000), (5000, 10000), (10000, 20000)]  # Lower and upper frequencies of each frequency bin

leds_per_column = 44

max_value_transformed_fft_data = 80

weighting = [2,8,8,16,16,32,32,64]  # scaling factors of frequency bins. rule of thumb: double frequency -> half power

spectrum_levels = [0, 0, 0, 0, 0, 0, 0, 0]  # this is the output

# Return power array index corresponding to a particular frequency
def get_power_array_index_of_frequency(frequency):
    return int(channels * chunk_size * frequency / sampling_frequency)


def calculate_spectrum_levels(data_chunk):
    global spectrum_levels
    # apply fft
    fft_data = np.fft.rfft(data_chunk)
    # remove last elementin array to make it the same size as the chunk
    fft_data = np.delete(fft_data, len(fft_data) - 1)
    # transform complex fft data to real data
    fft_data =np.abs(fft_data)
    # divide fft data into the frequency bins
    for i, frequency_bin in enumerate(frequency_bins):
        low_index = get_power_array_index_of_frequency(frequency_bin[0])
        high_index = get_power_array_index_of_frequency(frequency_bin[1])
        spectrum_levels[i] = np.mean(fft_data[low_index:high_index])  # TODO: use bell curve instead of mean (Flori)

    #tidy up spectrum level values
    spectrum_levels = np.divide(np.multiply(spectrum_levels, weighting), 1000000)  # TODO: 1e6 determined empirically
    spectrum_levels = np.interp(spectrum_levels, [0, max_value_transformed_fft_data], [0, leds_per_column])  # TODO: max_value_transformed_fft_data was determined empirically

    return spectrum_levels





####
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyaudio
import wave
from struct import unpack

wavfile = wave.open(r'C:\Users\giuse\Documents\SpectrumAnalyzer\test_music.wav', 'r')
sample_rate = wavfile.getframerate()
no_channels = wavfile.getnchannels()


py_audio = pyaudio.PyAudio()
stream = py_audio.open(format=py_audio.get_format_from_width(wavfile.getsampwidth()),
                       channels=wavfile.getnchannels(),
                       rate=wavfile.getframerate(),
                       output=True)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, ylim=(0, max_value_transformed_fft_data))

def animate(i):
    data = wavfile.readframes(chunk_size)
    # Convert raw data (ASCII string) to numpy array
    data = unpack("%dh" % (len(data) / 2), data)
    data = np.array(data, dtype='h')
    matrix = calculate_spectrum_levels(data)
    ax.clear()
    ax.set_ylim(0, max_value_transformed_fft_data)
    ax.bar(range(len(matrix)), matrix)
    stream.write(data)

ani = animation.FuncAnimation(fig, animate, interval=1)
plt.show()

print('done')



