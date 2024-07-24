#!/bin/bash

prep_yaml() {
    FUNC_CLASS=$1
    MODEL_NAME=$2

    TEST_YAML="<<: *$FUNC_CLASS
steps: 5
model: *$MODEL_NAME
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
    SOURCE_PATH=conf/include/tasks.yml

    # extract all FCs with root-level anchor def's as "task: &TASK_NAME\n"
    FUNC_CLASSES="$(sed -n 's/^task: &\(.*\)$/\1/p' $SOURCE_PATH)"

    echo -e "Detected the following function classes:\n$FUNC_CLASSES"
    while read func; do 
        run_check "$func" "small_gpt2"
    done <<< "$FUNC_CLASSES"
}


run_model_checks() {
    SOURCE_PATH=conf/include/models/base.yml

    # extract all models with root-level anchor definitions as "model: &MODEL_NAME\n"
    MODELS="$(sed -n 's/^model: &\(.*\)$/\1/p' $SOURCE_PATH)"

    echo -e "Detected the following models:\n$MODELS"
    while read model; do
        run_check "linear_regression" "$model"
    done <<< "$MODELS"
}
