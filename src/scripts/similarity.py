import json
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import CosineSimilarity
import sys
import numpy as np

args = sys.argv[1:]
name = args[0].replace('_',' ')
N = int(args[1])

int_to_card = json.load(open('././output/int_to_card.json','r'))
int_to_card = {int(k):v for k,v in int_to_card.items()}
card_to_int = {v:k for k,v in int_to_card.items()}

num_cards = len(int_to_card)

model = load_model('ml_files/deck_final')

cards = np.zeros((num_cards,num_cards))
np.fill_diagonal(cards,1)

dist_f = CosineSimilarity()

embs = model.encoder(cards)
idx = card_to_int[name]

dists = np.array([
    dist_f(embs[idx],x).numpy() for x in embs
])

ranked = dists.argsort()
print(f"\ncards most similar to {name}\n")
for i in range(1,N):
    card_idx = ranked[i]
    print("\t" + str(i) + ":",int_to_card[card_idx])#,dists[card_idx]))