# Figure index

[简体中文](README.zh-CN.md)

The collection includes 31 final charts from the exploratory analyses, modeling, and four follow-up questions, plus a public preview of the original poster. Download PNGs for any chart or choose the listed SVG or PDF where available.

The figures use the Kaggle v10 Craigslist dataset. Prices are advertised in USD. See the associated notebooks for sample definitions and methods.

## Exploratory summaries

[Notebook 03](../notebooks/03_price_age_mileage.ipynb) · [Notebook 04](../notebooks/04_categories_and_states.ipynb)

| Figure | Download |
| --- | --- |
| Main and full asking-price ranges | [PNG](exploratory/01_price_ranges.png) |
| Asking-price density by age and mileage | [PNG](exploratory/02_age_mileage_density.png) |
| Asking prices by condition | [PNG](exploratory/03_price_by_condition.png) |
| Asking prices by fuel | [PNG](exploratory/04_price_by_fuel.png) |
| Asking prices by vehicle type | [PNG](exploratory/05_price_by_vehicle_type.png) |
| Asking prices by manufacturer | [PNG](exploratory/06_price_by_manufacturer.png) |
| Median asking price by state | [PNG](exploratory/07_state_median_prices.png) |

## Initial model comparison

[Notebook 05](../notebooks/05_adjusted_associations_and_prediction.ipynb)

| Figure | Download |
| --- | --- |
| Prediction errors on the later holdout | [PNG](model-baseline/01_holdout_model_comparison.png) |
| Adjusted age and mileage contrasts | [PNG](model-baseline/02_adjusted_associations.png) |

## Q1 · Price distribution and anomalies

[Notebook 06](../notebooks/06_price_distribution_and_anomalies.ipynb)

| Figure | Download |
| --- | --- |
| Full and main price distributions | [PNG](q1/01_price_distribution.png) · [PDF](q1/01_price_distribution.pdf) |
| Tail probabilities | [PNG](q1/02_tail_probabilities.png) · [PDF](q1/02_tail_probabilities.pdf) |
| Heaping at common prices | [PNG](q1/03_price_heaping.png) · [PDF](q1/03_price_heaping.pdf) |
| Screening rates by vehicle type | [PNG](q1/04_vehicle_type_screening.png) · [PDF](q1/04_vehicle_type_screening.pdf) |
| Peer-relative anomaly screening | [PNG](q1/05_peer_relative_screen.png) · [PDF](q1/05_peer_relative_screen.pdf) |

## Q2 · Age and mileage

[Notebook 07](../notebooks/07_age_mileage_nonlinearity.ipynb)

| Figure | Download |
| --- | --- |
| Unadjusted age and mileage patterns | [PNG](q2/01_raw_age_mileage.png) · [SVG](q2/01_raw_age_mileage.svg) |
| Overlap comparisons for older vehicles | [PNG](q2/02_older_age_overlap.png) · [SVG](q2/02_older_age_overlap.svg) |
| Adjusted age and mileage profiles | [PNG](q2/03_adjusted_profiles.png) · [SVG](q2/03_adjusted_profiles.svg) |
| Age and mileage interaction | [PNG](q2/04_age_mileage_interaction.png) · [SVG](q2/04_age_mileage_interaction.svg) |
| Conditional quantile predictions | [PNG](q2/05_quantile_predictions.png) · [SVG](q2/05_quantile_predictions.svg) |

## Q3 · Vehicle group comparisons

[Notebook 08](../notebooks/08_comparable_vehicle_groups.ipynb)

| Figure | Download |
| --- | --- |
| Price distributions across vehicle groups | [PNG](q3/01_group_distributions.png) · [SVG](q3/01_group_distributions.svg) |
| Raw and overlap-weighted contrasts | [PNG](q3/02_overlap_weighted_contrasts.png) · [SVG](q3/02_overlap_weighted_contrasts.svg) |
| Price distributions by condition | [PNG](q3/03_condition_distributions.png) · [SVG](q3/03_condition_distributions.svg) |
| Condition price gaps by age | [PNG](q3/04_condition_gap_by_age.png) · [SVG](q3/04_condition_gap_by_age.svg) |
| Condition coverage across vehicle groups | [PNG](q3/05_condition_missingness.png) · [SVG](q3/05_condition_missingness.svg) |

## Q4 · State price differences

[Notebook 09](../notebooks/09_state_price_differences.ipynb)

| Figure | Download |
| --- | --- |
| Raw medians and model-adjusted state gaps | [PNG](q4/state_price_maps.png) · [PDF](q4/state_price_maps.pdf) |
| State ranks before and after adjustment | [PNG](q4/state_rank_changes.png) · [PDF](q4/state_rank_changes.pdf) |
| State price differences and intervals | [PNG](q4/state_intervals.png) · [PDF](q4/state_intervals.pdf) |
| Sensitivity across models and time windows | [PNG](q4/state_gap_sensitivity.png) · [PDF](q4/state_gap_sensitivity.pdf) |

## Rolling model validation

[Notebook 10](../notebooks/10_rolling_validation_and_error_analysis.ipynb)

| Figure | Download |
| --- | --- |
| Model error across validation windows | [PNG](validation/rolling_validation_mae.png) · [PDF](validation/rolling_validation_mae.pdf) |
| Model error by asking price and age | [PNG](validation/price_and_age_error_profile.png) · [PDF](validation/price_and_age_error_profile.pdf) |
| Model error after removing feature groups | [PNG](validation/feature_ablation.png) · [PDF](validation/feature_ablation.pdf) |
