# The site's throughput / pbatch prompts: anton-mel/stats synthetic.py prompt_words_for_tokens(N)
# (copied: seed 20260730, its word list, one word a token) plus common.py's unique suffix
# " (Request #i)"; the bench wraps each in the system + user chat template.
import json
import random
import sys

SEED = 20260730
WORDS = (
    "the of and to in a is that for it as was with be by on not he this are or his from at which but have an "
    "had they you were their one all we can her has there been if more when will would who so no she what "
    "about up out many then them these some him time into only its two could other new than first any my "
    "now such like our over man me even most made after also did much before here through years where "
    "must way well down should because each just those people how too little state good very make world "
    "still own see men work long get life never day another know while last might great old year off "
    "come since against go came right used take three states himself few house use during without again "
    "place american around however home small found mrs thought went say part once general high upon "
    "school every don does got united left number course war until always away something fact though "
    "water less public put think almost hand enough far took head yet government system better set told "
    "nothing night end why called didn eyes find going look asked later knew point next program city"
).split()


def words(n_tokens, salt=""):
    rng = random.Random(f"{SEED}:{salt}:{n_tokens}")
    return " ".join(rng.choice(WORDS) for _ in range(n_tokens))


n_tokens, count, salt, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
body = words(n_tokens, salt)
json.dump([f"{body} (Request #{i})" for i in range(count)], open(out, "w"))
