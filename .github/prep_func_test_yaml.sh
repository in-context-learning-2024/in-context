#!/bin/bash

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
