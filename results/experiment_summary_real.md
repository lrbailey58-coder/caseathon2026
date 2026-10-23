## Results: final unseen week W6 (previously unseen; models trained on W1-W5)

| Data | Countermeasure | Avg error (all) | Avg error (cargo) | 95th pct (all) | Data thrown out | Handed to human | Major corruptions caught | Damage undone (cargo) |
|---|---|---|---|---|---|---|---|---|
| Clean | none | 668 m | 533 m | 2,423 m | 0.0% | 0.0% | - | - |
| Clean | gate + human hand-off | 668 m | 544 m | 2,439 m | 1.9% | 15.6% | - | - |
| Stale | none | 693 m | 591 m | 2,344 m | 0.0% | 0.0% | - | - |
| Stale | gate + human hand-off | 669 m | 548 m | 2,424 m | 22.3% | 16.0% | 95% | 74% |
| Noisy | none | 687 m | 572 m | 2,434 m | 0.0% | 0.0% | - | - |
| Noisy | gate + human hand-off | 668 m | 549 m | 2,394 m | 17.5% | 16.6% | 55% | 60% |
| Both | none | 718 m | 642 m | 2,399 m | 0.0% | 0.0% | - | - |
| Both | gate + human hand-off | 669 m | 551 m | 2,403 m | 31.2% | 16.4% | 80% | 84% |

Physics-only baseline: 923 m average error.

### Is it luck? 95% ranges for the gaps (cargo ships, meters)

| Data | Damage from poisoning | Gained back by the gate | Left over vs clean |
|---|---|---|---|
| Stale | +42 to +78 | +29 to +60 | +8 to +23 |
| Noisy | +24 to +60 | +11 to +39 | +7 to +25 |
| Both | +71 to +159 | +59 to +137 | +9 to +28 |

Cost of the gate on clean data (cargo): +6 to +18 m.

- Error numbers include every prediction, also the ones handed to a human.
- Single training run; LightGBM has no randomness here, so variability comes from which voyages were tested.
- Tanker numbers are omitted: W6 has only 3 tanker voyages.
