# coding: utf-8

from audio_neural_model import neural_audio_style_transfer

def main():

	content = "inputs/lemons.mp3"
	style = "inputs/beat.mp3"

	nnet = neural_audio_style_transfer(content, style)
	nnet.optimize(alpha=0.01)
	nnet.save(output='outputs/lemons_beat.wav')



if __name__ == '__main__':

	main()