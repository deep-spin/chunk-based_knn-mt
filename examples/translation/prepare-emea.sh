#!/bin/bash

# wget http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz

# Usage: bash prepare-domadap.sh medical

DATADIR=$1
SOURCE_LANG=$2
TARGET_LANG=$3

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
SPM_PATH=/home/pam/sentencepiece/build/src/
SPM_MODEL=/media/hdd1/pam/mt/models/mbart50.ft.nn/sentence.bpe.model

if [ ! -d "$SCRIPTS" ]; then
  echo "Please set SCRIPTS variable correctly to point to Moses scripts."
  exit
fi

mkdir ${DATADIR}/processed

filesource=${DATADIR}/train.${SOURCE_LANG}
filetarget=${DATADIR}/train.${TARGET_LANG}

cat $filesource | \
  perl $NORM_PUNC $l | \
  perl $REM_NON_PRINT_CHAR | \
  #perl $TOKENIZER -threads 8 -a -l ${SOURCE_LANG}  >> ${DATADIR}/processed/train.tok.${SOURCE_LANG}

cat $filetarget | \
  perl $NORM_PUNC $l | \
  perl $REM_NON_PRINT_CHAR | \
  #perl $TOKENIZER -threads 8 -a -l ${TARGET_LANG}  >> ${DATADIR}/processed/train.tok.${TARGET_LANG}

$SPM_PATH/spm_encode --model=$SPM_MODEL < ${DATADIR}/train.${SOURCE_LANG} > ${DATADIR}/processed/train.spm.${SOURCE_LANG}
$SPM_PATH/spm_encode --model=$SPM_MODEL < ${DATADIR}/train.${TARGET_LANG} > ${DATADIR}/processed/train.spm.${TARGET_LANG}


for split in dev test
do

  #cat $filesource | \
  #  perl $TOKENIZER -threads 8 -a -l ${SOURCE_LANG}  >> ${DATADIR}/processed/${split}.tok.${SOURCE_LANG}

  #cat $filetarget | \
  #  perl $TOKENIZER -threads 8 -a -l ${TARGET_LANG}  >> ${DATADIR}/processed/${split}.tok.${TARGET_LANG}

  $SPM_PATH/spm_encode --model=$SPM_MODEL < ${DATADIR}/${split}.${SOURCE_LANG} > ${DATADIR}/processed/${split}.spm.${SOURCE_LANG}
  $SPM_PATH/spm_encode --model=$SPM_MODEL < ${DATADIR}/${split}.${TARGET_LANG} > ${DATADIR}/processed/${split}.spm.${TARGET_LANG}
done
