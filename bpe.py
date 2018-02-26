from math import log


class BPE(object):

    def __init__(self, vocab_file):
        with open(vocab_file, encoding="utf8") as f:
            self.words = [l.split()[0] for l in f]
            log_len = log(len(self.words))
            self.wordcost = {
                k: log((i+1) * log_len)
                for i, k in enumerate(self.words)}
            self.maxword = max(len(x) for x in self.words)

    def encode(self, s):
        """Uses dynamic programming to infer the location of spaces in a string
        without spaces."""

        s = s.replace(" ", "▁")

        # Find the best match for the i first characters, assuming cost has
        # been built for the i-1 first characters.
        # Returns a pair (match_cost, match_length).
        def best_match(i):
            candidates = enumerate(reversed(cost[max(0, i - self.maxword):i]))
            return min(
                (c + self.wordcost.get(s[i-k-1:i], 9e999), k+1)
                for k, c in candidates)

        # Build the cost array.
        cost = [0]
        for i in range(1, len(s) + 1):
            c, k = best_match(i)
            cost.append(c)

        # Backtrack to recover the minimal-cost string.
        out = []
        i = len(s)
        while i > 0:
            c, k = best_match(i)
            assert c == cost[i]
            out.append(s[i-k:i])

            i -= k

        return " ".join(reversed(out))


if __name__ == "__main__":
    bpe = BPE("en.wiki.bpe.op25000.vocab")
    print(bpe.encode(' this is our house in boomchakalaka'))
    # >>> ▁this ▁is ▁our ▁house ▁in ▁boom ch ak al aka
