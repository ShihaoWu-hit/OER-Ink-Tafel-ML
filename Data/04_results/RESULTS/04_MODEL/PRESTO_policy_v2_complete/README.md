# PRESTO_policy_v2_complete

This is the sole active model container selected by the user on 2026-07-30.
It is an operational three-class policy, not an independently confirmed model.

## Components

- `probability_engine/`: immutable PRESTO high-gate + direct-Gower
  conditional probability engine. Tree SHA256:
  `257c7438545007a3eada169a8d2cdbcccb59a30c3b2f89803128001cf422c0db`.
- `policy_v2/`: immutable verified policy-v2 layer. It inherits the parent
  probabilities exactly and changes only the high-alert action rule. Tree
  SHA256:
  `67005476961140e6d6b8a610b700c55863465f35953c426d5b9d698d4bb04f13`.

The parent engine is a required component of the final workflow, not a second
active model choice.

## What the model predicts

Each of the 97 records has a posterior-soft target over three contiguous
apparent-slope regions:

- `low_60_side`: below 84.8528 mV dec-1;
- `120_side`: 84.8528 through 144 mV dec-1;
- `high_off`: above 144 mV dec-1.

Validation uses 10 x 5 connected-group outer CV over 80 connected groups. One
record remains one sample; posterior draws are not expanded into pseudo-rows.

## Inputs

The final engine has nine logical model inputs, reported one-by-one in the
feature-importance summary:

1. catalyst family;
2. catalyst loading;
3. electrolyte molarity;
4. rotation rate;
5. scan rate;
6. two-state iR compensation status;
7. electrolyte scope;
8. solvent polarity coordinate;
9. ionomer / solvent ratio.

The polarity coordinate is one actual model input computed fold-locally from
surface tension, density, and best-available dielectric. Reporting those three
raw properties as if they were independent fitted features would describe a
different model.

## Headline metrics

For the inherited probability engine: soft log loss `0.819451`, multiclass
Brier `0.272466`, modal accuracy `0.711340`, and posterior-expected hard
accuracy `0.672611`.

For the selected policy-v2 action rule: modal accuracy `0.742268`, balanced
accuracy `0.577708`, macro F1 `0.602513`, posterior-expected hard accuracy
`0.706129`, high recall `0.500000`, high precision `0.666667`, high false
positive rate `0.022472`, and 6 high alerts among 97 records.

## Required limitation

The frozen validation report marks this run as global post-selection
development with `independent_confirmation=false` and
`scientifically_releasable=false`. The workflow may be used as the project's
selected operational result, but these fields must not be rewritten into an
independent validation claim. Associations are predictive and between-group;
they are not causal or mechanistic evidence.

