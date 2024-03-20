import zlib
import numpy as np
import matplotlib.pyplot as plt

def compress_string(input_string):
    """Compress a string and return the size of the compressed data."""
    compressed_data = zlib.compress(input_string.encode('utf-8'))
    return len(compressed_data)

def generate_random_string(length, alphabet):
    """Generate a random string of a given length from the specified alphabet."""
    return ''.join(np.random.choice(list(alphabet), length))

# Define different levels of randomness by specifying alphabets
alphabets = [
    'A',  # Least random: all characters are the same
    'AB',  # Low randomness: two possible characters
    'ABCDEFG',
    string.ascii_letters,  # Moderate randomness: all letters
    string.printable,  # Higher randomness: letters, digits, punctuation
]

length_of_string = 100000
compression_ratios = []

for alphabet in alphabets:
    random_string = generate_random_string(length_of_string, alphabet)
    compressed_size = compress_string(random_string)
    compression_ratio = compressed_size/len(random_string.encode('utf-8'))
    compression_ratios.append(compression_ratio)

# Plotting the compression ratios
plt.plot(compression_ratios, marker='o')
plt.title('Compression Ratio vs  (random string len=100000)')
plt.ylabel('Compression Ratio')
plt.xlabel('Randomness Level')
plt.xticks(range(len(alphabets)), ['A', 'AB', 'ABCDEFG', 'Letters', 'Printable'], rotation=45)
plt.grid(True)
plt.show()
