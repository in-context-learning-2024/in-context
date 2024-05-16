#!/bin/bash

prep_yaml() {
    FUNC_CLASS=$1
    FUNC_KWARGS=""

    MODEL_NAME=$2
    MODEL_KWARGS="
    n_positions: 21
    n_layer: 1
    n_embd: 4
    n_head: 1
"

    TEST_YAML="optim:
    type: adam
    lr: 0.0001
loss_fn:
    type: squared
function_class: 
    type: $FUNC_CLASS$FUNC_KWARGS
b_size: 8
seq_len: 20
x_dim: 10
x_dist:
    type: normal
steps: 5
model:
    type: $MODEL_NAME$MODEL_KWARGS
"

    echo -e "$TEST_YAML" > conf/test.yml
}

cut_file() {
    FILE_PATH=$1
    START_REGEX=$2
    END_REGEX=$3

    START_LINE=$(($(grep -n "$START_REGEX" "$FILE_PATH" | cut -f1 -d:)+1))
    END_LINE=$(($(grep -n "$END_REGEX" "$FILE_PATH" | cut -f1 -d:)-1))
    echo -e "$(sed -n "$(echo $START_LINE),$(echo $END_LINE) p" $FILE_PATH)"
}

run_check() {
    FUNC_NAME=$1
    MODEL_NAME=$2

    prep_yaml "$FUNC_NAME" "$MODEL_NAME"
    WANDB_MODE=offline WANDB_SILENT=true conda run -n in-context-learning \
        python src/ -c conf/test.yml
    if [[ $? != 0 ]]; then return 1; fi

    echo -e "Passed check with model \"$MODEL_NAME\"" \
          "\n     and function class \"$FUNC_NAME\""
}

run_func_checks() {
    SOURCE_PATH=src/function_classes/__init__.py

    LINES=$(cut_file $SOURCE_PATH "FUNCTION_CLASSES: dict" "}")
    # pull out what appears inside of the first pair of quotes on each line
    FUNC_CLASSES=$(echo "$LINES" | cut -d\" -f2 | sed -e '/^$/,$d')

    echo -e "Detected the following function classes:\n$FUNC_CLASSES"
    while read func; do 
        run_check "$func" "gpt2"
    done <<< "$FUNC_CLASSES"
}


run_model_checks() {
    SOURCE_PATH=src/models/__init__.py

    LINES=$(cut_file $SOURCE_PATH "MODELS: dict" "}")
    # pull out what appears inside of the first pair of quotes on each line
    MODELS=$(echo "$LINES" | cut -d\" -f2 | sed -e '/^$/,$d')

    echo -e "Detected the following models:\n$MODELS"
    while read model; do
        run_check "linear regression" "$model"
    done <<< "$MODELS"
}
