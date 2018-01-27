# coding: utf-8

from audio_neural_model import neural_audio_style_transfer
from plot import plot_spectrum

def main():

	#Hypeparameters and training sets
	content_name = "bach"				# content filename without suffix(.mp3)
	style_name = "beat"					# style filename
	alpha = 0.01  						# Larger alpha means more content in the output and alpha=0 means no content
	#####################################
	result_name = content_name+"_"+style_name
	
	content = "inputs/{0}.mp3".format(content_name)
	style = "inputs/{0}.mp3".format(style_name)


	nnet = neural_audio_style_transfer(content, style)
	a_content = nnet.a_content
	a_style = nnet.a_style
	plot_spectrum(a_content,filename=content_name)
	plot_spectrum(a_style,filename=style_name)
	

	nnet.optimize(alpha=alpha)
	nnet.save(output='outputs/{0}.wav'.format(result_name))
	a_result = nnet.a_result

	plot_spectrum(a_result,filename=result_name)



if __name__ == '__main__':

	main()