#!/bin/bash

# wget http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz

# Usage: bash prepare-domadap.sh medical

DATADIR=$1
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
BPECODES=/media/hdd1/pam/mt/models/wmt19.de-en/bpecodes
VOCAB=/media/hdd1/pam/mt/models/wmt19.de-en/dict.en.txt

if [ ! -d "$SCRIPTS" ]; then
  echo "Please set SCRIPTS variable correctly to point to Moses scripts."
  exit
fi

mkdir ${DATADIR}/processed

filede=${DATADIR}/train.de
fileen=${DATADIR}/train.en

cat $filede | \
  perl $NORM_PUNC $l | \
  perl $REM_NON_PRINT_CHAR | \
  perl $TOKENIZER -threads 8 -a -l de  >> ${DATADIR}/processed/train.tok.de

cat $fileen | \
  perl $NORM_PUNC $l | \
  perl $REM_NON_PRINT_CHAR | \
  perl $TOKENIZER -threads 8 -a -l en  >> ${DATADIR}/processed/train.tok.en

$FASTBPE/fast applybpe ${DATADIR}/processed/train.bpe.de ${DATADIR}/processed/train.tok.de $BPECODES $VOCAB
$FASTBPE/fast applybpe ${DATADIR}/processed/train.bpe.en ${DATADIR}/processed/train.tok.en $BPECODES $VOCAB

perl $CLEAN -ratio 1.5 ${DATADIR}/processed/train.bpe de en ${DATADIR}/processed/train.bpe.filtered 1 250

for split in dev test
do
  filede=${DATADIR}/${split}.de
  fileen=${DATADIR}/${split}.en

  cat $filede | \
    perl $TOKENIZER -threads 8 -a -l de  >> ${DATADIR}/processed/${split}.tok.de

  cat $fileen | \
    perl $TOKENIZER -threads 8 -a -l en  >> ${DATADIR}/processed/${split}.tok.en

  $FASTBPE/fast applybpe ${DATADIR}/processed/${split}.bpe.de ${DATADIR}/processed/${split}.tok.de $BPECODES $VOCAB
  $FASTBPE/fast applybpe ${DATADIR}/processed/${split}.bpe.en ${DATADIR}/processed/${split}.tok.en $BPECODES $VOCAB
done
