# SimpleTTS

A simple (and hopefully somewhat novel) approach to Text-to-Speech (TTS).

Dependencies:
```bash
# Must be installed for phonemizer
brew install espeak

# Might need to run this every time
export PATH="/opt/homebrew/bin:$PATH"
python your_script.py
```
## Architecture

1.  ~~**Text → phonemes**~~  ✅
    - Convert input text into phonetic tokens.  
        - Example:  
    - `"Hello world"` → `[HE, LO, SPACE, WO, R, LD]`

2. **Transformer encoder**  
   - Pass tokens through a multi-layer Transformer to produce contextual embeddings:  
   - `[embedding_1, embedding_2, ...]`

3. **Per-token prediction**  
   - For each embedding, predict:
       - a small spectrogram patch (fixed window)
       - a duration (distance from previous token)

4. **Time construction**  
   - Build timestamps by cumulative sum of durations:  
   - `t_i = t_{i-1} + d_i`

5. **Rendering**  
   - Place each token’s spectrogram patch into a global spectrogram at its predicted time.  
   - Overlapping patches are summed (with linear interpolation for training stability).

6. **(Optional) refinement**  
   - A Conv1D network can refine the final spectrogram.

7. ~~**Spectrogram → audio**~~  ✅
   - Convert to waveform using:
       - Griffin-Lim, or  
       - pretrained HiFi-GAN, or  
       - a separately trained model

## Key idea

Each token produces:
- a local sound patch
- a relative time offset

Speech is formed by overlapping and summing these token-level sound events.



### Data 

Lots of data exists. I am starting with CMU Arctic, A dataset of a number of people reading novels, split into single sentences

I am usiing the speaker EEY, since they sound the most emotive and normal

https://huggingface.co/datasets/MikhailT/cmu-arctic/tree/main/data


### Neural vocoder

For decoding I am using vocos, since this is lightweight and SOTA. Other models could be used, however you would need to match the mel spectrogram parameters

https://gemelo-ai.github.io/vocos/#audio-reconstruction-from-bark-tokens

