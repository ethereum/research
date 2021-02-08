echo -e "WIDTH_BITS\tWIDTH\tNUMBER_INITIAL_KEYS\tNUMBER_KEYS_PROOF\taverage_depth\tproof_size\tproof_time\tcheck_time" > stats.txt


python verkle_trie.py 5 65536 500 >> stats.txt
python verkle_trie.py 6 65536 500 >> stats.txt
python verkle_trie.py 7 65536 500 >> stats.txt
python verkle_trie.py 8 65536 500 >> stats.txt
python verkle_trie.py 9 65536 500 >> stats.txt
python verkle_trie.py 10 65536 500 >> stats.txt
python verkle_trie.py 11 65536 500 >> stats.txt
python verkle_trie.py 12 65536 500 >> stats.txt
python verkle_trie.py 13 65536 500 >> stats.txt
python verkle_trie.py 14 65536 500 >> stats.txt
python verkle_trie.py 15 65536 500 >> stats.txt
python verkle_trie.py 16 65536 500 >> stats.txt
