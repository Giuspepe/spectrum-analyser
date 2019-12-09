import numpy as np
from struct import unpack
import pyaudio
import logging
logging.basicConfig(level=logging.DEBUG)

chunk_size = 4096  # Use a multiple of 8 (FFT will compute faster with a multiple of 8)


py_audio = pyaudio.PyAudio()
device_index = 0
device_info = py_audio.get_device_info_by_index(device_index)
logging.debug(f'Using device {device_index}: {device_info})')
sampling_frequency = int(device_info['defaultSampleRate'])  # 44100 Hz
channels = device_info['maxInputChannels']  # 1
audio_stream = py_audio.open(format=pyaudio.paInt16, channels=channels, rate=sampling_frequency,
                              input=True, input_device_index=device_index)


frequency_bins = [(0, 156), (156, 313), (313, 625), (625, 1250), (1250, 2500),
                  (2500, 5000), (5000, 10000), (10000, 20000)]  # Lower and upper frequencies of each frequency bin

leds_per_column = 44

max_value_transformed_fft_data = 80

weighting = [2,8,8,16,16,32,32,64]  # scaling factors of frequency bins. rule of thumb: double frequency -> half power


# Return power array index corresponding to a particular frequency
def get_power_array_index_of_frequency(frequency):
    return int(channels * chunk_size * frequency / sampling_frequency)


def calculate_spectrum_levels():
    spectrum_levels = list()
    audio_data = audio_stream.read(chunk_size, exception_on_overflow=False)
    audio_data = unpack("%dh" % (chunk_size / 2), audio_data)
    audio_data = np.array(audio_data, dtype='h')
    # apply fft
    fft_data = np.fft.rfft(audio_data)
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



