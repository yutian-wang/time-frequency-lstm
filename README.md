# time-frequency-lstm
fundamental frequency estimation using tflstm in Keras

this work firstly process the audio spectrum into overlaped sequence with frequency axis, and then with time axis. the reresult of the preprocess is 4D data structure (time-batch, time-chunk-len, frequency-batch, frequency-chunk-len). to satisfy the lstm input restriction, the TimeDistributed wrapper is used on the first lstm layer. the values of the spectrum amplitude is scaled into (0,1).
On the other hand, the label sequence is discretized into the frequency bins, which is sample_rate/frame_size. In mir-1k corpus，the frequency bin interval is 25Hz. Then these discretized values are encoded into non-sparse one-hot codes. The number of the units of the output is the width of the one-hot codes.
the architecture of the networks is shown as below

Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 8, 17, 64)         0         
_________________________________________________________________
time_distributed_2 (TimeDist (None, 8, 256)            328704    
_________________________________________________________________
lstm_6 (LSTM)                (None, 8, 256)            525312    
_________________________________________________________________
lstm_7 (LSTM)                (None, 8, 256)            525312    
_________________________________________________________________
lstm_8 (LSTM)                (None, 256)               525312    
_________________________________________________________________
dense_2 (Dense)              (None, 16)                4112      
=================================================================

after 50 epoch training, the model gets 93.6% accuracy
