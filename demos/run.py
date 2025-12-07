from audiocraft.models import MusicGen
import torchaudio

print('Loading model...')
model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(duration=10)

prompts = ["soft piano with gentle rain",
"calming synth under forest ambience",
"acoustic guitar with ocean waves",
"loud percussion with light wind",
"lo-fi piano mixed with city street noise",
"slow violin over crackling fire",
"relaxing flute with birds chirping",
"deep clarinet with thunder rumbling",
"simple keyboard melody with waterfalls",
"minimal electronic beat with nighttime crickets"]


# prompts = [
#    "Midwest emo pop",
#    "indie kpop",
#    "lofi hiphop"
# ]
for i, prompt in enumerate(prompts):
    print(f'\nGenerating: {prompt}')
    wav = model.generate([prompt])
    torchaudio.save(f'traceySounds{i}.wav', wav[0].cpu(), model.sample_rate)
    print(f'Saved prompt_{i}.wav')

print('\nDone!')