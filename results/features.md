# Candidate generation and feature engineering

Partitions used: `train` (80 trips) and `student_test` (10 trips). Evaluator data is never used.
All numbers below come from `results/miss_rate.csv`, `results/alignment_summary.csv` and the
build output of `python -m src.candidates.build_all`.

## Why a candidate ranker
Every trip has its own road network (90 different networks), so an arc ID cannot be a class.
Each row is one (GPS point, nearby arc) pair and the model scores how likely that arc is the
road the vehicle was on. The per-point prediction is the arc with the highest score.

## Projection
Trips are spread across the world (trip centres from longitude -100 to 137, latitude -35 to
62; 82 of 90 trips lie outside UTM zone 32). Each trip is projected with its own local
azimuthal equidistant projection centred on its GPS track, so distances in metres stay
accurate at trip scale (a trip spans at most about 1.1 degrees of longitude).

## Candidate generation
For each GPS point, arcs within R metres (point-to-segment distance, spatial index built once
per trip) are candidates; the K nearest are kept. Chosen: R = 100 m, K = 12.

| R (m) | K=5 | K=8 | K=12 | K=16 |
|---|---|---|---|---|
| 30 | 4.22% | 3.13% | 2.87% | 2.78% |
| 50 | 3.30% | 2.10% | 1.80% | 1.69% |
| 75 | 2.87% | 1.63% | 1.31% | 1.19% |
| 100 | 2.57% | 1.30% | 0.94% | 0.82% |

Candidate miss rate (reference arc absent from the candidate set), over train + student_test.
R = 100 m, K = 12 gives 0.94% with about 8.9 candidates per point. R = 150 m gives identical
values because points with no route arc within 100 m have no reference arc and are excluded
(see below), so values above 100 m cannot be measured with this labelling.

## Features (`configs/features.yaml`)
| Feature | Why |
|---|---|
| `dist_m` | GPS noise is small relative to road spacing, so the true road is usually close |
| `offset_frac` | Where the point projects along the arc; distinguishes arcs the point only touches at an end |
| `arc_len_m` | Very short arcs (junction pieces) behave differently from long ones |
| `heading_diff_deg` | Angle between the GPS motion direction and the arc direction; separates the two directions of a two-way road |
| `speed_mps` | Speed from neighbouring points (only within 30 s); helps tell moving vehicles from stops |
| `rank_dist` | Relative closeness among this point's candidates |
| `dist_gap_best` | How much farther this arc is than the nearest candidate |
| `n_cands` | Local road density (junction or parallel roads) |
| `has_twin` | Whether the reverse arc exists (two-way road) |

Heading and speed use neighbouring points only if they are at most 30 s away; otherwise the
point is marked `after_gap`. When heading is undefined (no usable neighbour, or under 0.5 m of
motion) `heading_diff_deg` is set to 90 (neutral).

## Labels (assumption)
`.route` is only an ordered list of arcs, with no per-point labels. The reference arc for each
GPS point is derived by monotone alignment: points are matched to route positions that never go
backwards, by dynamic programming on point-to-arc distance. If no route arc is within 100 m the
point has no reference arc and is dropped from the candidate tables. The reference route is not
independently verified ground truth, so the miss rate measures coverage of the aligned route.

## Data-quality notes
- Train: 208,417 GPS points, 20,552 (9.9%) without a reference arc; mostly train trips 54
  (14.7% of points matched) and 33 (49% matched). Student test: 17 of 19,160 points.
- Large time gaps exist in some trips (up to 153,064 s); speed and heading are not computed
  across gaps.
- The candidate stage never reads `.route` (checked by `tests/test_no_leakage.py`).
