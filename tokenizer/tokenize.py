from random import shuffle
import sentencepiece as spm


def train(input_file, vocab_size, model_name, model_type, character_coverage,sentence_size):
    """
    search on https://github.com/google/sentencepiece/blob/master/doc/options.md to learn more about the parameters
    :param input_file: one-sentence-per-line raw corpus file. No need to run tokenizer, normalizer or preprocessor.
                       By default, SentencePiece normalizes the input with Unicode NFKC.
                       You can pass a comma-separated list of files.
    :param vocab_size: vocabulary size, e.g., 8000, 16000, or 32000
    :param model_name: output model name prefix. <model_name>.model and <model_name>.vocab are generated.
    :param model_type: model type. Choose from unigram (default), bpe, char, or word.
                       The input sentence must be pretokenized when using word type.
    :param character_coverage: amount of characters covered by the model, good defaults are: 0.9995 for languages with
                               rich character set like Japanse or Chinese and 1.0 for other languages with
                               small character set.
    :param sentence_size: amount of sentences that are trained in the tokenizer due to the size of the datatset.
    """
    input_argument = '--input=%s --model_prefix=%s --vocab_size=%s --model_type=%s --character_coverage=%s --input_sentence_size=%s' \
                     '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 '
    cmd = input_argument % (input_file, model_name, vocab_size, model_type, character_coverage,sentence_size)
    spm.SentencePieceTrainer.Train(cmd)


def run():

    input = '../data/description.txt'
    vocab_size = 8000
    model_name = 'des'
    model_type = 'bpe'
    character_coverage = 1
    sentence_size = 795443
    train(input, vocab_size, model_name, model_type, character_coverage,sentence_size)

    input = '../data/solutions.txt'
    vocab_size = 8000
    model_name = 'sol'
    model_type = 'bpe'
    character_coverage = 1
    sentence_size = 1500000
    train(input, vocab_size, model_name, model_type, character_coverage,sentence_size)


def test():
    sp = spm.SentencePieceProcessor()
    # text = "For each test case output a line containing a single integer, equal to the minimal possible number of Johnny's lies during the game."
    text = "from collections import Counter\n minr = 1\n"
    # sp.Load("./des.model")
    sp.Load("./sol.model")
    print(sp.EncodeAsPieces(text))
    print(sp.EncodeAsIds(text))
    # a = [12907, 277, 7419, 7318, 18384, 28724]
    # print(sp.decode_ids(a))


if __name__ == "__main__":
    # run()
    test()
