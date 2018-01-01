import numpy as np
import matplotlib.pyplot as plt 

def plot_spectrum(data,filename='Unknown'):
	plt.figure(figsize=[5,5])
	plt.title(filename)
	plt.imshow(data[:400,:])
	plt.savefig("figure/" +filename)
	print('{0} is plotted successfully!'.format(filename))


