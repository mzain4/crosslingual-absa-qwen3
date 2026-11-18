# Food UABSA checkpoint trajectory

Verified 14 November 2026. All 192 saved predictions across the three checkpoints reproduce their saved raw precision, recall and F1. Sentence text, source indices and gold outputs match the same 32 English and 32 Swahili development reviews. Every condition has zero format failures.

| Checkpoint | English raw F1 | Swahili raw F1 | Swahili omitted nonempty reviews |
|---|---:|---:|---:|
| Epoch 1, update 160 | 47.62 | 17.02 | 14/25, 56% |
| Epoch 2, update 320 | 52.46 | 5.26 | 21/25, 84% |
| Epoch 3, update 480 | 60.00 | 0.00 | 22/25, 88% |

The second-epoch checkpoint loses 7.54 English F1 points relative to the third epoch while retaining only 5.26 Swahili F1. It does not supply the desired balance on this development cohort. From the first to third epoch, English F1 rises by 12.38 points and Swahili F1 falls by 17.02. The increase in Swahili omissions accompanies that decline.

This trajectory supports investigating how English task adaptation changes transfer behavior. These observations alone cannot establish catastrophic forgetting or its mechanism. The checkpoint comparison uses one training seed and an already-inspected development cohort. The checkpoints come from one three-epoch learning-rate schedule.

The subsequent matched Food UABSA seed-13 run with learning rate 1e-4 has completed. It scores English 56.67 and Swahili 0.00 F1, compared with 60.00 and 0.00 for the original 2e-4 three-epoch control. The lower-rate configuration supplies no observed Swahili F1 gain on this cohort. [Verified lower-rate comparison](food_lower_lr_comparison.md)
