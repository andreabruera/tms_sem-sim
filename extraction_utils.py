
def check_present_words(args, rows, vocab):
    present_words = list()
    for w in rows:
        ### for fasttext in german we only use uppercase!
        if w[0].isupper() == False and args.lang=='de':
            if args.model=='fasttext':
                #or 'lm' in args.model or 'llama' in args.model:
                continue
        try:
            vocab.index(w)
        except ValueError:
            continue
        present_words.append(w)
    return present_words

def load_context_model(args):
    print('loading {}'.format(args.model))
    model = args.model.split('_')[0]
    layer = [int(args.model.split('-')[-1])]

    base_folder = os.path.join(
                                'collect_word_sentences',
                                'llm_vectors',
                                args.lang, 
                                'wac',
                                model,
                                )
    assert os.path.exists(base_folder)
    vocab = list()
    model = dict()
    for f in os.listdir(base_folder):
        if 'tsv' not in f:
            continue
        with open(os.path.join(base_folder, f)) as i:
            for l_i, l in enumerate(i):
                if l_i == 0:
                    continue
                line = l.strip().split('\t')
                word = line[0]
                l = int(line[1])
                if l in layer:
                    if word in model.keys():
                        model[word] = numpy.average([model[word].copy(), numpy.array(line[2:], dtype=numpy.float64)], axis=0)
                    else:
                        model[word] = numpy.array(line[2:], dtype=numpy.float64)
                    vocab.append(word)

    return model, vocab

def load_static_model(args):
    print('loading {}'.format(args.model))
    base_folder = os.path.join(
                                '/',
                                'data',
                                'u_bruera_software',
                                'word_vectors', 
                                args.lang, 
                                )
    if args.model == 'fasttext':
        model = fasttext.load_model(
                                    os.path.join(
                                        base_folder,
                                        'cc.{}.300.bin'.format(args.lang)
                                        )
                                    )
        vocab = model.words
    model = {w : model[w] for w in vocab}
    vocab = [w for w in vocab]

    return model, vocab

def load_context_surpr(args):
    print('loading {}'.format(args.model))
    model = args.model.split('_')[0]
    base_folder = os.path.join(
                                'collect_word_sentences',
                                'llm_surprisals',
                                args.lang, 
                                model,
                                )
    assert os.path.exists(base_folder)
    vocab = set()
    model = dict()
    for f in os.listdir(base_folder):
        if 'tsv' not in f:
            continue
        if 'sentence' in f:
            continue
        if args.dataset not in f:
            continue
        with open(os.path.join(base_folder, f)) as i:
            print(f)
            for l_i, l in enumerate(i):
                if l_i == 0:
                    continue
                line = l.strip().split('\t')
                word_one = line[0].strip()
                word_two = line[1].strip()
                s = float(line[2])
                #s = float(line[3])
                try:
                    model[word_one][word_two] = s
                except KeyError:
                    model[word_one] = {word_two : s}
                vocab.add(word_one)
                vocab.add(word_two)
    vocab = list(vocab)
    if len(vocab) < 10:
        raise RuntimeError()

    return model, vocab

def check_present_words(args, rows, vocab):
    present_words = list()
    for w in rows:
        ### for fasttext in german we only use uppercase!
        if w[0].isupper() == False and args.lang=='de':
            if args.model=='fasttext':
                #or 'lm' in args.model or 'llama' in args.model:
                continue
        try:
            vocab.index(w)
        except ValueError:
            continue
        present_words.append(w)
    return present_words
