# Environmental MusicGen: Fine-Tuned Hybrid Audio Generation
Collin Shen, Stanley Ou, Jim Zhou  
10-423/623/723 Generative AI Course Project, Carnegie Mellon University  
December 6, 2025

## 1. Abstract
This project explores text-conditioned generation of environmental–instrumental hybrid audio using transformer-based generative AI. Instead of general symbolic or music-only synthesis, we focus on adapting Meta’s pretrained MusicGen model to produce audio that combines musical structure with natural ambience (e.g., “lo-fi piano over rain,” “calming synth with forest ambience”). MusicGen relies on EnCodec, a residual vector quantization (RVQ) audio tokenizer, and an autoregressive transformer trained on instrument-only music; therefore, it struggles to generate environmental textures.

We fine-tune MusicGen using LoRA adapters and environmental datasets such as UrbanSound8K and ESC-50. We evaluate the pretrained baseline and future fine-tuning runs using CLAP text–audio similarity, with additional metrics (FAD, KL divergence).

## 2. Introduction
Music generation remains a complex multimodal challenge due to the hierarchical, stochastic, and temporally structured nature of audio signals. While recent breakthroughs such as MusicGen and MusicLM enable text-to-music generation with impressive musical coherence, these models are typically trained on clean, instrument-only datasets. As a result, they lack the ability to synthesize audio that blends musical and environmental components.

Our project aims to bridge this gap by adapting MusicGen to generate environmental–instrumental hybrid audio clips from text prompts such as “soft piano with light rainfall” or “ambient synth under city street noise.” This task requires the model to capture both the harmonic patterns of music and the aperiodic, noisy characteristics of real-world ambience.

In addition to scientific value, hybrid audio generation is practically relevant to film, gaming, and ambience-based content creation. Existing generative music models struggle with these domains due to their bias toward harmonic signals.

## 3. Dataset / Task

### Datasets
We shift from symbolic music datasets to environmental sound datasets:

- UrbanSound8K – 8,732 labeled environmental sounds (sirens, drilling, children playing, etc.)
- ESC-50 – 2,000 clips across 50 environmental categories (rain, wind, fire, forest, etc.)

Each environmental clip is paired with a natural-language caption generated from its label (e.g., “steady rain on pavement,” “children playing in a park”).

We may additionally construct synthetic hybrid audio by mixing environmental clips with instrumental stems from FMA:

x_hybrid = 0.5 * x_instrumental + 0.5 * x_environment

These hybrid audios serve as references for later FAD and KL divergence evaluation.

### Task Definition
Given a textual description T, the model generates an audio clip A containing both musical content and environmental ambience. Because pretrained MusicGen has no exposure to environmental audio, this task is expected to be challenging without fine-tuning.

### Evaluation Metrics
- Frechet Audio Distance (FAD) – perceptual similarity  
- CLAP Similarity – cosine similarity  
- KL Divergence – token distribution comparison  
- Human Evaluation – qualitative assessment  

## 4. Related Work
(omitted here for brevity, full text included in final md file)

## 5. Approach
(omitted for brevity; full text is included in the final file)

## 6. Experiments
(Baseline tables and details included)

## 7. Plan
(Project phases, timeline, infrastructure)

## 8. Thought-Experiment on Compute
(GPU usage, budget expansion discussion)

## 9. References
(Full reference list included)
