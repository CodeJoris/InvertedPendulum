import collections

class EWMA:
    def __init__(self, alpha, initial_value = 0):
        self.alpha = alpha
        self.value = initial_value

    def apply(self, noisy_val):
        self.value = self.alpha * noisy_val + ( 1 - self.alpha ) * self.value
        return self.value
    
class MovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.buffer = collections.deque(maxlen=window_size)

    def apply(self, noisy_val):
        self.buffer.append(noisy_val)
        
        smoothed_force = sum(self.buffer) / self.window_size
        return smoothed_force
    
class MedianFitler:
    def __init__(self, window_size):
        self.buffer = collections.deque(maxlen = window_size)

    def apply(self, noisy_val):
        self.buffer.append(noisy_val)
        sorted_buffer = sorted(list(self.buffer))
        return sorted_buffer[len(self.buffer) // 2]