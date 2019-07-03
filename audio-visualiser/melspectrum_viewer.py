import python_speech_features as psf
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import sys

(rate,sig) = wav.read(sys.argv[1])

# window_size = int(np.ceil(0.5*rate))
# overlap = int(np.ceil(0.25*rate))

window_size = 10000
overlap = 5000

print(rate)
print(sig.size)
print(window_size)
print(overlap)

mfcc_feat = psf.mfcc(sig, rate)
fbank_feat = psf.logfbank(sig,rate)
plt.imshow(fbank_feat.T, aspect='auto')
plt.show()

for i in range(0, sig.size, overlap):
    print(i)
    fbank_feat = psf.logfbank(sig[i:i + window_size], rate)
    mfcc_feat = psf.mfcc(sig[i:i+window_size], rate)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.imshow(fbank_feat.T, aspect='auto')
    ax1.set_title('Mel-filterbank energy features')
    ax1.set_xlabel('Frames')
    ax1.set_ylabel('Filters')

    ax2 = fig.add_subplot(212)
    ax2.imshow(mfcc_feat.T, aspect='auto')
    ax2.set_title('MFCC features')
    ax2.set_xlabel('Frames')
    ax2.set_ylabel('Cepstrum Index')

    #ax3 = fig.add_subplot(313)
    #ax3.bar(np.arange(0,mfcc_feat.shape[1]), height=mfcc_feat[1])
    plt.show()

