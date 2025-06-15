# Subword Representations and Pre-trained Language Model for Thai Part-of-Speech Tagging in Universal Part of Speech Scheme

## Abstract

Part-of-speech tagging is a crucial step for many downstream tasks; yet, it is understudied for Thai text. Thai words can be composed by a few subwords, which have their own meanings, despite being an isolating language. Therefore, subword representation should provide helpful cues for POS tagging in Thai. We mapped the Thai-specific ORCHID tag set to the multilingual applicable Universal Dependency tag set and experiment with character, syllable, and byte-pair encoding (BPE) as subword features which can form word embeddings consumed by taggers. Our results reveal that syllables representations are better than word and character embeddings for Thai POS tagging, especially for handling out of vocabulary words. And BERT models, which use BPE and pretrained language models, yield the state-of-the art results.

## Introduction

Part-of-speech tagging (POS tagging) is an important task in natural language processing (NLP) because it can help in tasks such as syntactic parsing or named entity recognition. This task is well-studied for high-resource languages such as English and Chinese, yet under-explored for Thai. Modern approaches includes neural networks where word representation is composed from a character representation or otherwise pretrained. (Bohnet et al., 2018)  (Akbik et al., 2018) A major challenge in Thai POS tagging is that Thai is an isolating language, so inflectional morphology cannot guide the tagger. A word can be composed by juxtaposing a few subwords.

For example, กองปราบปราม is a noun denoting the Crime Suppression Division. The word can be further split into กอง (division) and ปราบปราม (suppress). And the ปราบปราม verb compound can be further segmented into ปราบ and ปราม . Therefore, a Thai POS tagger should further segment the word and use the subwords as features. We hypothesize that if a word is segmented into syllables, then syllable features should improve the performance.

Recently, the paradigm of pre-training and fine-tuning model such as ELMo (Peters et al., 2018) and BERT (Devlin et al., 2018) has received much attention. This paradigm uses wordpiece representation to prevent out-of-vocabulary problem and learn the representation for potential infixes. We hypothesize that this approach suits Thai POS tagging because noun and verb compounds can be captured by the wordpieces. In this study, we explore syllable and wordpiece representation for Thai in handling out of vocabularies problems and improving the overall performance of Thai POS tagging.

To the best of our knowledge, the ORCHID corpus has been the only Thai corpus annotated with POS tags (Sornlertlamvanich et al., 1970). The tagset consists of 42 tags designed specifically to model the Thai language. This tagset has never been mapped to a more manageable Universal Dependency (UD) (Nivre et al., 2020) tagset, which can be used across many different languages. We aim to extend the existing Thai POS tagging studies by converting ORCHID tagset to UD tagset, which is more suitable for downstream tasks such as dependency parsing.

Our contributions can be summarized as follows:

• We show that subword-level representations, namely syllable, character, and byte-pair encoding (BPE) are better than word embedding for POS tagging.

• We present the BERT-based state-of-the-art model for Thai POS tagging against many strong neural baselines.

• We mapped the widely-used ORCHID tagset to the new standard Universal POS tagset.

## Related Works

Neural networks with character-based representations have been extensively studied in English POS tagging. (Akbik et al., 2018) proposed an approach that uses character-level language models to learn context-dependent word representations. Also, (Bohnet et al., 2018) combined pretrained word embeddings with character-based word encodings to gain a sentence-level contextsensitive encoding. Both studies used their respective learned representations with Bi-LSTM network and achieved top performances in in Penn POS tagging benchmark.

For Thai POS tagging, the only study that implement neural architectures is done by (Boonkwan and Supnithi, 2018), which also used character n-gram embeddings with backoff with Bi-GRU models for determining word boundary as well as part of speech, with good results in handling out-of-vocabulary words.

## Mapping from Orchid tagset to UD POS tagset

We map the 42-tag ORCHID tagset to 13-tag Universal Dependency (UD) tagset. UD tagset has been proven to be effective for dependency parsing and interoperable across languages. The conversion to UD also reduces POS-tagging model complexity. We found that most ORCHID tags can be mapped to exactly one UD tag.

Two UD tags are worth elaborating. First, NUM category in UD comes from nominal and determiner tags in ORCHID. We found tags that are used with numerical words: cardinal number (NCNM), ordinal number (NONM), determiner with cardinal number expression (DCNM), and determiner with ordinal number form (DONM). As UD annotation guideline states that NUM can function as nouns as well as determiners, but ordinal numbers are not tagged with NUM, (contributors, ) we decided to map NCNM and DCNM to NUM, NONM to NOUN, and DONM to DET. Second, no ORCHID tag is mapped to ADJ. This is because in Thai, verbs and adjectives has the same contextual distributions, which means they can be categorized as the same syntactic category. Words that can be thought of as ADJ when translated into English as adjectives (for example, สวย, beautiful ) are attributive verbs. Therefore, ADJ tags do not exist in our annotation. Although Thai has no inflectional morphology, many verb and noun compounds exist, especially in ORCHID corpus. Syllables have been shown to help with word segmentation because they memorize some patterns of subwords in Thai text (Chormai et al., 2019). Byte-pair-encoding (BPE) is also used to reduce vocabulary size and handle out of vocabulary problem. We hypothesize that character patterns, syllables, and BPE help in analyzing sub-component of out-of-vocabulary words. To test this hypothesis, we use these subword representation as features of CRF and BiLSTM-CRF, which are known to work well with POS tagging in other languages. we use syllable unigrams because most Thai words have fewer than three syllables. Syllables are generated by automatically segmenting words into orthographical pronounceable syllables.

## Model Description and Experimental Setup

$$
Character ห|อ|ก|า|ร|ค| ้ |า|จ| ั |ง|ห|ว| ั |ด|เ|ช| ี |ย|ง|ใ|ห|ม| ่ Syllable หอ|การ|ค้ า|จั ง|หวั ด|เชี ยง|ใหม่ BPE _หอ|การค้ า|จั งหวั ดเชี ยงใหม่
$$

In BiLSTM-CRF models, we used subword embeddings to compose a word embedding (Figure 1). For character features, we apply BiLSTM models to compose a word embedding from a sequence of character embeddings in that word. The same process is done for syllable features and BPE features. The hidden activation in the last timesteps should capture the subword representation of the whole word. For BPE features, we use pre-trained tokenizer and embeddings from BPEmb (Heinzerling and Strube, 2018) with vocabulary size at 20,000 and embedding vector dimension at 50. Then, we use BiLSTM models to compose a word from a sequence of BPEs. The composed word embeddings are then fed into another BiLSTM and CRF.

Orthographical Features For each model we added orthographical features as a base feature set. We use the following set of features in all models experimented to provide the models with orthographical information of each word token: 1.) If the word is all-Thai characters. 2.) If the word is all Roman characters. 3.) If the word is all numerical characters. (including Thai numerals) 4.) If the word contains numerical characters. 5.) If the word is all punctuation characters. 6.) If the word contains punctuation characters. These features are used in all of our models. These features should help in finding words with NUM and PUNCT labels, as well as capture non-Thai words that are mostly NOUNs and PROPNs. These features are concatenated with the word embeddings composed from the subword features or from pretrained language models.

Pretrained Language Model: We also experimented with BERT-based models, which work best with BPE representations. We use each model's pre-trained wordpiece tokenizer and embeddings, which is fine-tuned in the training process. We use the pretrained models and finetune them for this task. Similar to the BERT's original fine tuning method for NER (Devlin et al., 2018), we represent each word with the first wordpiece token in that word as an input to the model (Figure 1). We then pass the embeddings of each word from the last layer of the pretrained transformers models to a dropout layer, and then a fully connected layer with softmax activation to predict the POS. We experimented with three pretrained language models: Thai BERT (BERT-TH) from ThAIKeras project 1 , Multilingual BERT (mBERT) (Devlin et al., 2018), and XLM-RoBERTa (Conneau et al., 2019).   We evaluate each models's performance by examining both overall accuracy and out-of-vocabulary (words not found in the training set) accuracy on the test set. The results of CRF and BiLSTM-CRF models suggest that models with subword features performs better than non-compositional word embedding (CRF and BiLSTM-CRF+Word in Table 4) . Syllables features also generally performs better than character and BPE features in the same models.

## Results and Discussion

The monolingual BERT-TH model performs the best both in overall accuracy (0.96) and OOV accuracy (0.94). The OOV accuracy is higher than the second best model by a large margin. This illustrates that the BERT fine tuning approaches generalize well to both seen and unseen words. The multilingual BERT models perform much more poorly than the monolingual BERT-TH. This is possibly because the BPE vector space for the multilingual model is shared across languages, and this task does not benefit from the crosslingual transfer.

## Conclusion

We present a mapping scheme from ORCHID part-of-speech tags to UD part-of-speech tags. We found that the subword representation is crucial for Thai POS tagging by experimenting with composing word embeddings from character, syllable, and BPE sequences. Furthermore, we present the fine-tuned Thai BERT as the state-of-the art POS tagging model for the ORCHID dataset with UD tags, which outperform the other strong pretrained language model baselines.

Table 2 :

Table 1 :

Table 3 :

Table 4 :

Alan Akbik, Duncan Blythe, Roland Vollgraf, Contextual string embeddings for sequence labeling, 2018-08, Proceedings of the 27th International Conference on Computational Linguistics, Association for Computational Linguistics.

UNKNOWN, None, 2018, Morphosyntactic tagging with a meta-bilstm model over context sensitive token encodings, .

UNKNOWN, None, 2018, Bidirectional deep learning of context representation for joint word segmentation and pos tagging, .

UNKNOWN, None, 2019, Attacut: A fast and accurate neural thai word segmenter, .

UNKNOWN, None, 2019, Unsupervised cross-lingual representation learning at scale, .

UNKNOWN, None, 2018, BERT: pre-training of deep bidirectional transformers for language understanding, CoRR.

Benjamin Heinzerling, Michael Strube, ; , Khalid Choukri, Christopher Cieri, Thierry Declerck, Sara Goggi, Koiti Hasida, Hitoshi Isahara, Bente Maegaard, Joseph Mariani, Hélène Mazo, BPEmb: Tokenization-free Pre-trained Subword Embeddings in 275 Languages, 2018-05-07, Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), .

UNKNOWN, None, 2020, Universal dependencies v2: An evergrowing multilingual treebank collection, .

UNKNOWN, None, 2018, Deep contextualized word representations, .

UNKNOWN, None, 1970, Thai part-of-speech tagged corpus, .