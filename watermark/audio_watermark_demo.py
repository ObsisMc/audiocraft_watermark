from audiocraft.models import MusicGen
import soundfile as sf
import torch

from watermark.watermark_processor import WatermarkAudioDetector


def read_wav(path):
    audio = sf.read(path)
    return audio

def save_wav(audio, sr, name):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    
    # audio need to be two dimension (T, C) and T must be larger than 3
    if len(audio.shape) == 3:  # (B, ...)
        audio = audio[0]  # only get the first batch
    elif len(audio.shape) == 1:
        audio = audio[..., None]
    
    if audio.shape[1] > 3:  # (C, T)
        audio = audio.T
    
    sf.write(name, audio, sr)


def init_model(top_k=250, duration=30, watermark_model=False):
    model = MusicGen.get_pretrained('facebook/musicgen-medium')

    model.set_generation_params(
        use_sampling=True,
        top_k=top_k,
        duration=duration,
        watermark_mode=watermark_model
    )
    
    print(f"Number of codebooks: {model.compression_model.quantizer.total_codebooks}",
          f"Size of codebooks {model.compression_model.quantizer.bins}")
    return model


def init_detector(model):
    lm = model.lm
    encodec = model.compression_model

    detector = WatermarkAudioDetector(vocab=[i for i in range(lm.vocab_size)],
                                    gamma=lm.gamma_wm, # should match original setting
                                    seeding_scheme=lm.seeding_scheme_wm, # should match original setting
                                    device=model.device, # must match the original rng device type
                                    z_threshold=4.0,
                                    compression_model=encodec,
                                    layer_wm=lm.layer_wm
                                    # ignore_repeated_ngrams=True
                                    )
    return detector


def generate_audio(model):
    output = model.generate(
        descriptions=[
            #'80s pop track with bassy drums and synth',
            #'90s rock song with loud guitars and heavy drums',
            'Progressive rock drum and bass solo',
            #'Punk Rock song with loud drum and power guitar',
            # 'Bluesy guitar instrumental with soulful licks and a driving rhythm section',
            #'Jazz Funk song with slap bass and powerful saxophone',
            # 'drum and bass beat with intense percussions'
        ],
        progress=True, return_tokens=True
        )
    
    return output[0]

def detect_audio(model, audio_path=None, audio=None):
    assert (audio_path is not None) ^ (audio is not None)
    
    detector = init_detector(model)
    
    # load audio, the audio should be (C, T)
    if audio_path:
        audio, sr = read_wav(audio_path)  # only 1 channel audio
        audio = torch.tensor(audio[None, ...], dtype=torch.float32).to(model.device)
    else:
        # suppose audio is (B, C, T)
        if len(audio.shape) == 3:
            audio = audio[0]  # get the first audio in a batch
        
    
    score_dict = detector.detect(audio) # or any other text of interest to analyze
    info_print = '\n'.join([str(k) + ':' + str(v) for k, v in score_dict.items()])
    print(f"Detection results:\n{'='*20}\n",
          f"{info_print}",
          f"\n{'='*20}")
    
    return audio, score_dict
    

def watarmark_detect_audio(model):
    detector = init_detector(model)
    
    # generate audio
    with torch.no_grad():
        output = model.generate(
            descriptions=[
                # 'Bluesy guitar instrumental with soulful licks and a driving rhythm section',
                'Progressive rock drum and bass solo',
                ],
            progress=True, return_tokens=True
            )
        audio = output[0][0]  # (C, T)
    
    score_dict = detector.detect(audio) # or any other text of interest to analyze
    info_print = '\n'.join([str(k) + ':' + str(v) for k, v in score_dict.items()])
    print(f"Detection results:\n{'='*20}\n",
          f"{info_print}",
          f"\n{'='*20}")
    
    return audio, score_dict
    

def encode_decode_back(audio, compress_model):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    audio_tensor = torch.tensor(audio[None, None, ...], dtype=torch.float32).to(device)
    print(audio_tensor.shape)
    encode_audio = compress_model.encode(audio_tensor)
    audio_codes1 = encode_audio[0]
    print(audio_codes1.shape)

    decode_audio = compress_model.decode(audio_codes1, None)
    audio_restored = decode_audio

    encode_audio_back = compress_model.encode(audio_restored)
    audio_codes2 = encode_audio_back[0]

    acc = (audio_codes1 == audio_codes2).sum() / audio_codes2.numel()
    return acc, audio_codes1, audio_codes2


if __name__ == "__main__":
    ## 1. test1: generate audio
    # model = init_model(duration=8)
    # audio = generate_audio(model)
    # save_wav(audio, 32000, f"audio_generated_{model.duration}s.wav")
    
    ## 2. test2: test watermarking and detecting
    # duration = 15
    # model = init_model(duration=duration, watermark_model=True)
    # audio, score_dict = watarmark_detect_audio(model)
    # save_wav(audio, 32000, f"audio_watermarked_{duration}s.wav")
    
    ## 3. test3: compare results without and with watermark
    duration = 15
    model = init_model(duration=duration, watermark_model=False)
    audio = generate_audio(model)
    audio, score_dict = detect_audio(model, audio=audio)
    save_wav(audio, 32000, f"audio_generated_{model.duration}s.wav")
    
    model = init_model(duration=duration, watermark_model=True)
    audio, score_dict = watarmark_detect_audio(model)
    save_wav(audio, 32000, f"audio_watermarked_{duration}s.wav")
    
    ## 4. test4: load and detect audio
    # duration = 8
    # model = init_model(duration=duration, watermark_model=False)
    # audio_path = f"audio_watermarked_{duration}s.wav"
    # audio_path = f"audio_generated_{duration}s.wav"
    # detect_audio(model, audio_path=audio_path)
    
    
    ## 5. test5: test whether encodec can encode and decode lossless
    # model = init_model()
    # audio, sr = read_wav("test.wav")
    # acc, codes1, codes2 = encode_decode_back(audio, model.compression_model)
    # print(acc)
    # for i in range(codes1.shape[1]):
    #     print((codes1[0,i,:] == codes2[0,i,:]).sum() / codes2[0,i,:].numel())
    