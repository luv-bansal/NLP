import fastai
from fastai.text.all import *

path=untar_data(URLs.IMDB)

files=get_text_files(path)

get_imdb = partial(get_text_files, folders=['train', 'test', 'unsup'])
#Making dataset for Language Model
dls_lm = DataBlock(
    blocks=TextBlock.from_folder(path, is_lm=True),
    get_items=get_imdb, splitter=RandomSplitter(0.1)
).dataloaders(path, path=path, bs=128, seq_len=80)

dls_lm.show_batch(max_n=2)

learn = language_model_learner(
    dls_lm, AWD_LSTM, drop_mult=0.3, 
    metrics=[accuracy, Perplexity()]).to_fp16()

learn.fit_one_cycle(1, 2e-2)
learn.unfreeze()
learn.fit_one_cycle(10, 2e-3)

learn.save_encoder('finetuned')

TEXT = "I liked this movie because"
N_WORDS = 40
N_SENTENCES = 2
preds = [learn.predict(TEXT, N_WORDS, temperature=0.75) 
         for _ in range(N_SENTENCES)]
print("\n".join(preds))

#Making dataset for final Classsification task
dls_clas = DataBlock(
    blocks=(TextBlock.from_folder(path, vocab=dls_lm.vocab),CategoryBlock),
    get_y = parent_label,
    get_items=partial(get_text_files, folders=['train', 'test']),
    splitter=GrandparentSplitter(valid_name='test')
).dataloaders(path, path=path, bs=128, seq_len=72)

learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, 
                                metrics=accuracy).to_fp16()
learn = learn.load_encoder('finetuned')

learn.fit_one_cycle(1, 2e-2)

learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3))