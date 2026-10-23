## Results: final unseen week W6 (previously unseen; models trained on W1-W5)

| Data | Countermeasure | Avg error (all) | Avg error (cargo) | 95th pct (all) | Data thrown out | Handed to human | Major corruptions caught | Damage undone (cargo) |
|---|---|---|---|---|---|---|---|---|
| Clean | none | 677 m | 544 m | 2,462 m | 0.0% | 0.0% | - | - |
| Clean | gate + human hand-off | 680 m | 546 m | 2,485 m | 1.9% | 19.1% | - | - |
| Stale | none | 701 m | 586 m | 2,391 m | 0.0% | 0.0% | - | - |
| Stale | gate + human hand-off | 681 m | 551 m | 2,468 m | 22.7% | 19.1% | 94% | 84% |
| Noisy | none | 690 m | 562 m | 2,469 m | 0.0% | 0.0% | - | - |
| Noisy | gate + human hand-off | 676 m | 548 m | 2,442 m | 16.4% | 19.6% | 50% | 79% |
| Both | none | 715 m | 601 m | 2,428 m | 0.0% | 0.0% | - | - |
| Both | gate + human hand-off | 678 m | 550 m | 2,452 m | 31.0% | 19.0% | 78% | 89% |

Physics-only baseline: 1,005 m average error.

### Is it luck? 95% ranges for the gaps (cargo ships, meters)

| Data | Damage from poisoning | Gained back by the gate | Left over vs clean |
|---|---|---|---|
| Stale | +36 to +46 | +30 to +40 | +4 to +8 |
| Noisy | +13 to +24 | +9 to +20 | +2 to +6 |
| Both | +48 to +66 | +42 to +60 | +3 to +9 |

Cost of the gate on clean data (cargo): -0 to +3 m.

- Error numbers include every prediction, also the ones handed to a human.
- Single training run; LightGBM has no randomness here, so variability comes from which voyages were tested.
- Tanker numbers are omitted: W6 has only 9 tanker voyages.
