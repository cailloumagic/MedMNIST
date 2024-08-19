import platform

# Define base output directory paths based on the operating system
if platform.system() == 'Windows':
    base_output_dir = r'C:\Users\pierr\Desktop\Cours\BME4\Coding\Outputs'
else:  # Path for Linux
    base_output_dir = r'/home/ptreyer/Outputs'
