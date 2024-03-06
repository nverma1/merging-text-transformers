import random
from glob import glob
from tqdm import tqdm 
import sys

# pass path to bookscorpus
BOOKSROOT = sys.argv[1]
OUTPUT = 'sample0.txt'

# sample N_LINES_PER_FILE lines from each book
N_LINES_PER_FILE = 100


with open(OUTPUT, 'w+') as f_out:
    # books path should have structure BOOKSROOT/{genre}/{book}.txt
    for filename in tqdm(glob(BOOKSROOT + '*/*.txt')):
        with open(filename, 'r', encoding='utf-8',errors='ignore') as f:
            all_lines = list(filter(lambda x: x != '\n',f.readlines()))
            n_total_lines = len(all_lines)
            if n_total_lines >= N_LINES_PER_FILE:
                lines = random.sample(all_lines, N_LINES_PER_FILE)
                f_out.write(''.join(lines))
