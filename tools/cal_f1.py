def compute_f1(guessed_sentences, correct_sentences):
    prec = compute_precision(guessed_sentences, correct_sentences)
    rec = compute_precision(correct_sentences, guessed_sentences)

    f1 = 0
    if (prec+rec) > 0:
        f1 = 2.0 * prec *rec /(prec + rec)

    return prec, rec, f1

def compute_precision(guessed_sentences, correct_sentences):
    assert (len(guessed_sentences) == len(correct_sentences))
    correctCount = 0
    count = 0
    for sentenceIdx in range(len(guessed_sentences)):
        guessed = guessed_sentences[sentenceIdx]
        correct = correct_sentences[sentenceIdx]

        assert (len(guessed) == len(correct))
        idx = 0
        while idx < len(guessed):
            count += 1
            if guessed[idx] == correct[idx]:
                correctCount += 1
            idx += 1

    if count > 0:
        return float(correctCount) / count