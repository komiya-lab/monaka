#! /bin/bash

NODE_FORMAT="%m\t%f[9]\t%f[6]\t%f[7]\t%F-[0,1,2,3]\t%f[4]\t%f[5]\t%f[13]\t%f[26]\t%f[27]\t%f[28]\n"
UNK_FORMAT="%m\t%m\t%m\t%m\tUNK\t%f[4]\t%f[5]\t\n"
EOS_FORMAT="EOS\n"
BOS_FORMAT=""
MODEL="all_in_one"
OUTPUT_FORMAT="mecab"
SERVER="http://127.0.0.1:5000"
DIC="gendai"
SENTENCE[0]=""
IND=0

while (( $# > 0 ))
do
  case $1 in
    # ...
    --node-format | --node-format=*)
      if [[ "$1" =~ ^--node-format= ]]; then
        NODE_FORMAT=$(echo $1 | sed -e 's/^--node-format=//')
      elif [[ -z "$2" ]] || [[ "$2" =~ ^-+ ]]; then
        echo "'option' requires an argument." 1>&2
        exit 1
      else
        NODE_FORMAT="$2"
        shift
      fi
      ;;
    --unk-format | --unk-format=*)
      if [[ "$1" =~ ^--unk-format= ]]; then
        UNK_FORMAT=$(echo $1 | sed -e 's/^--unk-format=//')
      elif [[ -z "$2" ]] || [[ "$2" =~ ^-+ ]]; then
        echo "'option' requires an argument." 1>&2
        exit 1
      else
        UNK_FORMAT="$2"
        shift
      fi
      ;;
    --eos-format | --eos-format=*)
      if [[ "$1" =~ ^--eos-format= ]]; then
        EOS_FORMAT=$(echo $1 | sed -e 's/^--eos-format=//')
      elif [[ -z "$2" ]] || [[ "$2" =~ ^-+ ]]; then
        echo "'option' requires an argument." 1>&2
        exit 1
      else
        EOS_FORMAT="$2"
        shift
      fi
      ;;
    --bos-format | --eos-format=*)
      if [[ "$1" =~ ^--bos-format= ]]; then
        BOS_FORMAT=$(echo $1 | sed -e 's/^--bos-format=//')
      elif [[ -z "$2" ]] || [[ "$2" =~ ^-+ ]]; then
        echo "'option' requires an argument." 1>&2
        exit 1
      else
        BOS_FORMAT="$2"
        shift
      fi
      ;;
    --output-format | --output-format=*)
      if [[ "$1" =~ ^--output-format= ]]; then
        OUTPUT_FORMAT=$(echo $1 | sed -e 's/^--output-format=//')
      elif [[ -z "$2" ]] || [[ "$2" =~ ^-+ ]]; then
        echo "'option' requires an argument." 1>&2
        exit 1
      else
        OUTPUT_FORMAT="$2"
        shift
      fi
      ;;
    # ...
    --dic | --dic=*)
      if [[ "$1" =~ ^--dic= ]]; then
        DIC=$(echo $1 | sed -e 's/^--dic=//')
      elif [[ -z "$2" ]] || [[ "$2" =~ ^-+ ]]; then
        echo "'option' requires an argument." 1>&2
        exit 1
      else
        DIC="$2"
        shift
      fi
      ;;
    --model | --model=*)
      if [[ "$1" =~ ^--model= ]]; then
        MODEL=$(echo $1 | sed -e 's/^--model=//')
      elif [[ -z "$2" ]] || [[ "$2" =~ ^-+ ]]; then
        echo "'option' requires an argument." 1>&2
        exit 1
      else
        MODEL="$2"
        shift
      fi
      ;;
    --server | --server=*)
      if [[ "$1" =~ ^--server= ]]; then
        SERVER=$(echo $1 | sed -e 's/^--server=//')
      elif [[ -z "$2" ]] || [[ "$2" =~ ^-+ ]]; then
        echo "'option' requires an argument." 1>&2
        exit 1
      else
        SERVER="$2"
        shift
      fi
      ;;
    *)
      SENTENCE[IND]=\"$1\"
      IND=`expr $IND + 1`
      ;;
  esac
  shift
done

URL=${SERVER}/model/${MODEL}/dic/${DIC}/parse
sents="$(IFS=,; echo "${SENTENCE[*]}")"
JSON=`cat << EOS
{
  "sentence": [$sents],
  "node_format": "$NODE_FORMAT",
  "output_format": "$OUTPUT_FORMAT",
  "unk_format": "$UNK_FORMAT",
  "eos_format": "$EOS_FORMAT",
  "bos_format": "$BOS_FORMAT"
}
EOS`
echo $URL
echo $sents
echo "$JSON"

curl -X POST -H "Content-Type: application/json" -d "$JSON" $URL

#trap rm_tmpfile EXIT