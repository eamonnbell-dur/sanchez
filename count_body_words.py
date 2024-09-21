import os
import spacy
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import tqdm

nlp = spacy.load("en_core_web_sm")

directory = 'output'

body_words = [
    "hand", "eye", "mouth", "finger", "arm", "leg", "foot", "head", "ear", "nose",
    "shoulder", "knee", "elbow", "wrist", "ankle", "toe", "thumb", "back", "chest",
    "stomach", "hip", "neck", "throat", "chin", "forehead", "cheek", "lip", "tongue",
    "walk", "run", "jump", "sit", "stand", "bend", "stretch", "lift", "carry", "push",
    "pull", "touch", "grab", "hold", "shake", "wave", "nod", "blink", "wink", "smile",
    "frown", "chew", "bite", "swallow", "breathe", "cough", "sneeze", "yawn", "hug", "kiss"
]

body_words_lemmas = set([token.lemma_ for token in nlp(' '.join(body_words))])

def process_file(filename):
    if filename.endswith('.txt'):
        parts = filename.split('_')
        duration = parts[0]
        gender = parts[1]
        genre = parts[2]
        speciality = parts[3]
        attempt_number = parts[-1].split('.')[0].split('_')[-1]
        
        with open(os.path.join(directory, filename), 'r') as file:
            content = file.read()
            doc = nlp(content)
            words = [token.lemma_.lower() for token in doc if token.is_alpha]
            total_words = len(words)
            body_word_count = sum(1 for word in words if word in body_words_lemmas)
            proportion = body_word_count / total_words if total_words > 0 else 0
            
            return {
                'filename': filename,
                'body_words': body_word_count,
                'total_words': total_words,
                'proportion': proportion,
                'duration': duration,
                'gender': gender,
                'genre': genre,
                'speciality': speciality,
                'attempt_number': attempt_number
            }
    return None

results = []
with ProcessPoolExecutor() as executor:
    for result in tqdm.tqdm(executor.map(process_file, os.listdir(directory)), total=len(os.listdir(directory))):
        if result is not None:
            results.append(result)

df = pd.DataFrame(results)

df.to_pickle('body_word_proportions.pkl')

print("DataFrame saved as 'body_word_proportions.pkl'.")