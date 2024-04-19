#!/bin/bash

prep_yaml() {
    FC_NAME=$1
    KWARGS=""

    TEST_YAML="optim:
        type: adam
        lr: 0.0001
    loss_fn:
        type: squared
    function_class: 
        type: $FC_NAME$KWARGS
    b_size: 8
    seq_len: 20
    x_dim: 10
    x_dist:
        type: normal
    steps: 5
    model:
        type: gpt2
        n_positions: 21
        n_layer: 1
        n_embd: 4
        n_head: 1
    "

    echo -e "$TEST_YAML" > conf/test.yml
}

run_test() {
    TEST_NAME=$1
    conda run -n in-context-learning wandb offline
    bash .github/prep_func_test_yaml.sh \""$TEST_NAME"\"
    conda run -n in-context-learning python src/ -c conf/test.yml
}

run_all_tests() {
    START_LINE=$(grep -n "FUNCTION_CLASSES = {" src/function_classes/__init__.py  | cut -f1 -d:)
    END_LINE=$(grep -n "}" src/function_classes/__init__.py  | cut -f1 -d:)
    FUNC_CLASSES=$(sed -n '$START_LINE,$END_LINE p' src/function_classes/__init__.py)

    while read line; do 
        run_test \""$line"\"
    done <<< "$FUNC_CLASSES"
}
