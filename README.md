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
   - Start with sinusoidal positional encoding

3. **Per-token prediction**  
   - For each embedding, predict:
       - a small spectrogram patch (one fixed patch per token)
           - We could add additional transformer or convolutional layers here if needed
       - Absolute time position of this patch
           - Absolute position + monatonicity loss is used (rather than just a time difference between patches) so errors are not compounded from previous time errors (although this problem isn't completely removed)

4. **Rendering**  
   - Place each token’s spectrogram patch into a global spectrogram given its predicted time.  
   - Overlapping patches are linearly summed (with linear interpolation for training stability)
   - This forms a final mel spectrogram

5. **(Optional) refinement**  
   - A Conv1D network can refine the final spectrogram.
  
6. **Loss**
   - Loss is computed in two components
       - Spectrum loss: MAE (L1 loss) between predicted spectrum and real
           - Values below t=0 are removed
           - Values after t=t_max are penalised as if compared to silence
        - Monatonicity loss: We penalise patches for being out of order
        - We can adjust the relative weighting of each loss via a constant

8. ~~**Spectrogram → audio**~~  ✅
   - Convert to waveform using:
       - Vocos vocoder model
       - Or simple random phase assumption

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

