#!/bin/bash

set -euo pipefail

# Replace appropriate variables for paths and languages

REPO=/home/pam/chunked_knnmt


src_l="de"
tgt_l="en"
#url="https://wit3.fbk.eu/archive/2017-01-trnted/texts/de/en/en-de.tgz"
data="/media/hdd1/pam/mt/data/iwslt2017/de-en"
raw=$data/raw
prep=$data/processed
bin=$data/bin
src_dict="/media/hdd1/pam/mt/models/wmt19.$src_l-$tgt_l/dict.$src_l.txt"
tgt_dict="/media/hdd1/pam/mt/models/wmt19.$src_l-$tgt_l/dict.$tgt_l.txt"
#src_spm="/projects/tir5/users/patrick/data/paracrawl/en-fr/prep/spm.model"
#tgt_spm="/projects/tir5/users/patrick/data/paracrawl/en-fr/prep/spm.model"

#mkdir -p $raw $processed $bin

#archive=$raw/${url##*/}
#echo $archive
#if [ -f "$archive" ]; then
#    echo "$archive already exists, skipping download and extraction..."
#else
#    wget -P $raw $url
#    if [ -f "$archive" ]; then
#        echo "$url successfully downloaded."
#    else
#        echo "$url not successfully downloaded."
#        exit 1
#    fi#

#    tar --strip-components=1 -C $raw -xzvf $archive
#fi

echo "extract from raw data..."
#rm -f $data/*.${src_l}-${tgt_l}.*
#python ${REPO}/examples/translation/iwslt17/prepare_corpus.py $raw $data -s $src_l -t $tgt_l 

ln -sf $src_dict $prep/dict.${src_l}.txt
ln -sf $tgt_dict $prep/dict.${tgt_l}.txt
#ln -sf $src_spm $prep/spm.${src_l}.model
#ln -sf $tgt_spm $prep/spm.${tgt_l}.model


#echo "applying sentencepiece model..."
#for split in "train" "valid" "test"; do 
#    for lang in $src_l $tgt_l; do 
#        python scripts/spm_encode.py \
#            --model $prep/spm.$lang.model \
#                < $data/${split}.${src_l}-${tgt_l}.${lang} \
#                > $prep/${split}.${src_l}-${tgt_l}.${lang}
#    done
#done

#echo "binarizing..."
#fairseq-preprocess \
#    --source-lang ${src_l} --target-lang ${tgt_l} \
#    --trainpref ${prep}/train.${src_l}-${tgt_l} --validpref ${prep}/valid.${src_l}-${tgt_l} --testpref ${prep}/test.${src_l}-${tgt_l} \
#    --srcdict ${prep}/dict.${src_l}.txt --tgtdict ${prep}/dict.${tgt_l}.txt \
#    --destdir ${bin} \
#    --workers 20


HOME=/home/pam
if [ -z $HOME ]
then
  echo "HOME var is empty, please set it"
  exit 1
fi
SCRIPTS=$HOME/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
FASTBPE=$HOME/fastBPE
BPECODES=/media/hdd1/pam/mt/models/wmt19.$src_l-$tgt_l/bpecodes

if [ ! -d "$SCRIPTS" ]; then
  echo "Please set SCRIPTS variable correctly to point to Moses scripts."
  exit
fi

filede=${data}/train.de-en.de
fileen=${data}/train.de-en.en

cat $filede | \
  perl $NORM_PUNC de | \
  perl $REM_NON_PRINT_CHAR | \
  perl $TOKENIZER -threads 8 -a -l de  >> ${data}/processed/train.tok.de

cat $fileen | \
  perl $NORM_PUNC en | \
  perl $REM_NON_PRINT_CHAR | \
  perl $TOKENIZER -threads 8 -a -l en  >> ${data}/processed/train.tok.en

$FASTBPE/fast applybpe ${data}/processed/train.bpe.de ${data}/processed/train.tok.de $BPECODES $src_dict
$FASTBPE/fast applybpe ${data}/processed/train.bpe.en ${data}/processed/train.tok.en $BPECODES $src_dict

#perl $CLEAN -ratio 1.5 ${data}/processed/train.bpe de en ${data}/processed/train.bpe.filtered 1 250

for split in dev test
do
  filede=${data}/${split}.de-en.de
  fileen=${data}/${split}.de-en.en

  cat $filede | \
    perl $TOKENIZER -threads 8 -a -l de  >> ${data}/processed/${split}.tok.de

  cat $fileen | \
    perl $TOKENIZER -threads 8 -a -l en  >> ${data}/processed/${split}.tok.en

  $FASTBPE/fast applybpe ${data}/processed/${split}.bpe.de ${data}/processed/${split}.tok.de $BPECODES $src_dict
  $FASTBPE/fast applybpe ${data}/processed/${split}.bpe.en ${data}/processed/${split}.tok.en $BPECODES $src_dict
done


echo "binarizing..."
fairseq-preprocess \
    --source-lang ${src_l} --target-lang ${tgt_l} \
    --trainpref ${prep}/train.bpe --validpref ${prep}/dev.bpe --testpref ${prep}/test.bpe \
    --srcdict ${prep}/dict.${src_l}.txt --tgtdict ${prep}/dict.${tgt_l}.txt \
    --destdir ${bin} \
    --workers 20


cp $data/*.docids $bin