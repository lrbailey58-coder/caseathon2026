## Results: final unseen week W6 (previously unseen; models trained on W1-W5)

| Data | Countermeasure | Avg error (all) | Avg error (cargo) | 95th pct (all) | Data thrown out | Handed to human | Major corruptions caught | Damage undone (cargo) |
|---|---|---|---|---|---|---|---|---|
| Clean | none | 667 m | 543 m | 2,454 m | 0.0% | 0.0% | - | - |
| Clean | gate + human hand-off | 671 m | 547 m | 2,487 m | 1.9% | 18.8% | - | - |
| Stale | none | 690 m | 582 m | 2,379 m | 0.0% | 0.0% | - | - |
| Stale | gate + human hand-off | 669 m | 546 m | 2,460 m | 22.6% | 18.8% | 95% | 92% |
| Noisy | none | 683 m | 566 m | 2,468 m | 0.0% | 0.0% | - | - |
| Noisy | gate + human hand-off | 667 m | 553 m | 2,444 m | 16.4% | 19.6% | 50% | 56% |
| Both | none | 707 m | 608 m | 2,415 m | 0.0% | 0.0% | - | - |
| Both | gate + human hand-off | 665 m | 546 m | 2,429 m | 30.9% | 18.9% | 78% | 96% |

Physics-only baseline: 995 m average error.

### Is it luck? 95% ranges for the gaps (cargo ships, meters)

| Data | Damage from poisoning | Gained back by the gate | Left over vs clean |
|---|---|---|---|
| Stale | +33 to +45 | +30 to +41 | +1 to +5 |
| Noisy | +17 to +30 | +7 to +19 | +7 to +13 |
| Both | +55 to +77 | +53 to +73 | -0 to +6 |

Cost of the gate on clean data (cargo): +2 to +7 m.

- Error numbers include every prediction, also the ones handed to a human.
- Single training run; LightGBM has no randomness here, so variability comes from which voyages were tested.
- Tanker numbers are omitted: W6 has only 9 tanker voyages.
