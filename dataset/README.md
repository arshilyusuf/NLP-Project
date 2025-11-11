# Dataset for Hindi Sarcasm Detection

## Dataset Format

The dataset should be a CSV file named `hindi_sarcasm_dataset.csv` with the following columns:

- `text`: The Hindi text (Devanagari or Romanized)
- `label`: 0 for non-sarcastic, 1 for sarcastic

## Example

```csv
text,label
वाह! क्या बात है, तुमने तो कमाल कर दिया!,1
आज मौसम बहुत सुहावना है,0
हमें तो ये पता ही नहीं था,1
मैं आज स्कूल जा रहा हूँ,0
```

## Adding Your Own Data

1. Create a CSV file with your data
2. Place it in the `dataset/` folder
3. Name it `hindi_sarcasm_dataset.csv`
4. Run `python train_model.py` to train on your data

## Sample Dataset

If no dataset is found, the training script will automatically create a sample dataset with 40 examples (20 sarcastic, 20 non-sarcastic) to get you started.

## Tips for Better Results

- Include diverse examples
- Balance sarcastic and non-sarcastic samples
- Include both Devanagari and Romanized Hindi
- Add examples with different sarcasm levels
- Include context-specific examples

