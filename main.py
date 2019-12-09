
import board
import time
import neopixel
import threading
import array
import random
from math import cos
import numpy as np
from struct import unpack
import pyaudio
import logging

# definitions

num_rows = 44  # Zeile
num_columns = 8
num_pixels_total = num_rows * num_columns
pixel_pin = board.D18
ORDER = neopixel.GRB
brightness = 255
levelColor = (0, 0, 200)
dotColor = (200, 0, 0)
spectrum_levels = [0, 0, 0, 0, 0, 0, 0, 0, ]


class LedStripe(threading.Thread):
    oldValue = [0, 0, 0, 0, 0, 0, 0, 0]
    #value = [0, 0, 0, 0, 0, 0, 0, 0]
    dotfallingrate = 0

    def __init__(self):
        threading.Thread.__init__(self)
        self.pixels = neopixel.NeoPixel(pixel_pin, num_pixels_total, brightness=1, auto_write=False, pixel_order=ORDER)
        self.oldValue = [0, 0, 0, 0, 0, 0, 0, 0]

    def run(self):
        #self.sinus(100)
           while True:
                #for index in range(0, 8):
                #    self.value[index] = random.randint(0, 43)  # grab new FFT values
                self.fallingDot()
                self.refreshStripe()
              #  time.sleep(0.1)

    def refreshStripe(self):
        for column in range(0, num_columns):
            for neglevel in range(0, num_rows - int(spectrum_levels[column])):
                self.pixels[num_rows * column + num_rows - 1 - neglevel] = (0, 0, 0)
            for level in range(0, int(spectrum_levels[column])):
                self.pixels[num_rows * column + level] = levelColor
            self.pixels[num_rows * column + self.oldValue[column]] = dotColor
        self.pixels.show()

    def fallingDot(self):
        for column in range(0, num_columns):
            if self.oldValue[column] <= int(spectrum_levels[column]):
                self.oldValue[column] = int(spectrum_levels[column])
            elif self.oldValue[column] > 0 and self.dotfallingrate > 2:
                self.oldValue[column] -= 1
        if self.dotfallingrate > 2:
            self.dotfallingrate = 0
        else:
            self.dotfallingrate += 1

    def sinus(self, iteration):
        value = [0,0,0,0,0,0,0,0]
        for index in range(0, iteration):
            for c in range(0, 25):
                for i in range(0, 8):
                    value[i] = translate(cos(i * 6.24 / 6 + c) * 100, -100, 100, 10, 34)
                    self.pixels[value[i] + i * num_rows] = (0, 0, 200)
                    self.pixels.show()
                    time.sleep(0.2)
                    self.pixels.fill((0, 0, 0))
                    self.pixels.show

    def translate(self, value, leftMin, leftMax, rightMin, rightMax):
        # Figure out how 'wide' each range is
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin

        # Convert the left range into a 0-1 range (float)
        valueScaled = float(value - leftMin) / float(leftSpan)

        # Convert the 0-1 range into a value in the right range.
        return rightMin + (valueScaled * rightSpan)


class FFT(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        logging.basicConfig(level=logging.DEBUG)
        self.chunk_size = 1024  # Use a multiple of 8 (FFT will compute faster with a multiple of 8)
        py_audio = pyaudio.PyAudio()
        device_index = 0
        device_info = py_audio.get_device_info_by_index(device_index)
        logging.debug(f'Using device {device_index}: {device_info})')
        self.sampling_frequency = int(device_info['defaultSampleRate'])  # 44100 Hz
        self.channels = device_info['maxInputChannels']  # 1
        self.audio_stream = py_audio.open(format=pyaudio.paInt16, channels=self.channels, rate=self.sampling_frequency,
                                     input=True, input_device_index=device_index)

        self.frequency_bins = [(0, 156), (156, 313), (313, 625), (625, 1250), (1250, 2500), (2500, 5000), (5000, 10000),
                          (10000, 20000)]  # Lower and upper frequencies of each frequency bin

        self.max_value_transformed_fft_data = 50
        self.weighting = [2, 8, 8, 16, 16, 32, 32, 64]  # scaling factors of frequency bins. rule of thumb: double frequency -> half power
# Return power array index corresponding to a particular frequency
    def get_power_array_index_of_frequency(self,frequency):
        return int(self.channels * self.chunk_size * frequency / self.sampling_frequency)


    def run(self):
        while True:
            global spectrum_levels
            audio_data = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
            audio_data = unpack("%dh" % (len(audio_data) / 2), audio_data)
            audio_data = np.array(audio_data, dtype='h')
            # apply fft
            fft_data = np.fft.rfft(audio_data)
            # remove last elementin array to make it the same size as the chunk
            fft_data = np.delete(fft_data, len(fft_data) - 1)
            # transform complex fft data to real data
            fft_data = np.abs(fft_data)
            # divide fft data into the frequency bins
            for i, frequency_bin in enumerate(self.frequency_bins):
                low_index = self.get_power_array_index_of_frequency(frequency_bin[0])
                high_index = self.get_power_array_index_of_frequency(frequency_bin[1])
                spectrum_levels[i] = np.mean(fft_data[low_index:high_index])  # TODO: use bell curve instead of mean (Flori)

            # tidy up spectrum level values
            spectrum_levels = np.divide(np.multiply(spectrum_levels, self.weighting), 1000000)  # TODO: 1e6 determined empirically
            spectrum_levels = np.interp(spectrum_levels, [0, self.max_value_transformed_fft_data],  [0, num_rows]) # TODO: max_value_transformed_fft_data was determined empirically

def main():
    try:
        stripe = LedStripe()
        stripe.start()
        FFTlocal = FFT()
        FFTlocal.start()

    except Exception as e:
        print("EXCEPTION CATCHED")
        print(e)


if __name__ == '__main__':
    main()

