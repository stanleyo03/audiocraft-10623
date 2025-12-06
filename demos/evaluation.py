import laion_clap
import torch
import glob

print('Loading CLAP model...')
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()  # downloads checkpoint automatically

# Your generated files and their prompts
audio_files = glob.glob('output_*.wav')
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

print('\n=== CLAP Evaluation ===\n')

for audio_file, prompt in zip(audio_files, prompts):
    print(f'File: {audio_file}')
    print(f'Prompt: "{prompt}"')
    
    # Get audio and text embeddings
    audio_embed = model.get_audio_embedding_from_filelist([audio_file])
    text_embed = model.get_text_embedding([prompt])
    
    # Calculate similarity (0-1, higher is better)
    similarity = torch.cosine_similarity(
        torch.tensor(audio_embed), 
        torch.tensor(text_embed)
    ).item()
    
    print(f'CLAP Score: {similarity:.4f}')
    print('-' * 50)

print('\n✅ Evaluation complete!')